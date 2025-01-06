import cplex
import logging
import sqlite3
import pandas as pd
import sot_fbm_staffing.util.readers as readers

from typing import Dict

from sot_fbm_staffing.formulation.model import BasicModel
from sot_fbm_staffing.formulation.short_term_formulation import ShortTermFormulation
from sot_fbm_staffing.data.helpers import OutputData, DataHolder, Process, TabulatedParameterHolder
from sot_fbm_staffing.general_configurations import DevelopDumping, Configuration, FileNames, InputOutputPaths
from sot_fbm_staffing.output_test import OutputTest
from sot_fbm_staffing.data.shift_parameters import ShiftParametersGenerator


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class RollingExecutor:
    def __init__(
            self,
            shift_interval: int,
            shift_params: ShiftParametersGenerator,
            data_holders: Dict[Process, DataHolder],
            params: Dict[Process, TabulatedParameterHolder],
            raw_eh_parameters: pd.DataFrame,
            df_dates: pd.DataFrame,  # <-- These are the human readable timestamps for the integer epochs.
            post_processors: Dict[Process, None] = {}
    ) -> None:

        self.shift_interval = shift_interval
        self.shift_params = shift_params
        self.data_holders = data_holders
        self.params = params
        self.raw_eh_parameters = raw_eh_parameters
        self.human_epoch = df_dates
        self.post_processors = post_processors
        self.previous_solution = []  # list of cpx solution values
        self.objective_value_staffing_optimization = 0
        self.objective_value_stock_anticipation = 0

    def last_epoch_for_shift(self, shift):
        epochs = self.shift_params.epochs_for_shift.get(shift)
        return epochs.max() if len(epochs) else None

    def max_cpt_for_epoch_range(self, start: int, end: int) -> int:
        if start is None or end is None:
            return None

        epochs = set(cpt for dh in self.data_holders.values() for cpt in dh.cpts_for_epoch_range(start, end))
        return max(epochs) if epochs else None

    @staticmethod
    def update_separated_vars(existing: OutputData, new: OutputData) -> None:
        """
        This method is part of the setup of the rolling--horizon heuristic, used in some prior
        versions and left as an available resource.

        Separate means that we deliver workers and processings separately.

        Should start with:
        base_case_separated_vars = OutputData({}, pd.DataFrame(), {}, pd.DataFrame())
        """
        existing.hourly_workers = existing.hourly_workers.append(new.hourly_workers)
        existing.polys_follow_up = existing.polys_follow_up.append(new.polys_follow_up)

        for process, data in new.workers.items():
            existing.workers[process] = existing.workers.get(process, pd.DataFrame()).append(data) 

        for process, data in new.processed.items():
            existing.processed[process] = existing.processed.get(process, pd.DataFrame()).append(data)

    @staticmethod
    def write_separate_results(separated_vars: OutputData):
        """
        separate means that we deliver workers and processings separately
        """
        for process, data in separated_vars.workers.items():
            data.to_csv(FileNames.WRKRS_OUTPUT % (InputOutputPaths.BASEDIR_OUT, process.name.lower()), index=False)

        for process, data in separated_vars.processed.items():
            data.to_csv(FileNames.ITEMS_OUTPUT % (InputOutputPaths.BASEDIR_OUT, process.name.lower()), index=False)

        if DevelopDumping.DEV and DevelopDumping.MAKE_TABLE:
            con = sqlite3.connect(FileNames.DATABASE_MODEL)
            for process, data in separated_vars.workers.items():
                data.to_sql(name='out_' + (FileNames.WRKRS_OUTPUT[3:][:-4] % process.name.lower()), con=con,
                                    if_exists='replace', index=False)
            for process, data in separated_vars.processed.items():
                data.to_sql(name='out_' + (FileNames.ITEMS_OUTPUT[3:][:-4] % process.name.lower()), con=con,
                                      if_exists='replace', index=False)

            separated_vars.polys_follow_up.to_sql(name='out_' + (FileNames.POLYS_OUTPUT[3:][:-4]),
                                            con=con,
                                            if_exists='replace', index=False)
            if Configuration.activate_hourly_workers:
                separated_vars.hourly_workers.to_sql(
                        name='out_' + (FileNames.HOURLY_WRKRS_OUTPUT[3:][:-4]),
                        con=con,
                        if_exists='replace', index=False)
                con.commit()
                con.close()

        separated_vars.polys_follow_up.to_csv(FileNames.POLYS_OUTPUT %
                                    InputOutputPaths.BASEDIR_OUT, index=False)

        if Configuration.activate_hourly_workers:
            separated_vars.hourly_workers.to_csv(FileNames.HOURLY_WRKRS_OUTPUT %
                                       InputOutputPaths.BASEDIR_OUT, index=False)

    def post_process(self, out_dfs: Dict[Process, pd.DataFrame]) -> Dict[Process, pd.DataFrame]:
        """
        This remains here awaiting the 'subcanalization' to be included in a roadmap.
        subcanalization == split the output by sub--carrier
        """
        if not self.post_processors:
            return out_dfs

        output = dict()
        for process, data in out_dfs.items():
            post_processor = self.post_processors.get(process)
            output[process] = post_processor.execute(data) if post_processor else data

        return output

    def run(self) -> Dict[Process, pd.DataFrame]:
        #TODO revisar porque esto asume un orden para definir la ultima hora valida
        shift_to = self.shift_interval - 1
        epoch_start = 0
        last_fixed_epoch = self.last_epoch_for_shift(shift_to)
        epoch_end = self.max_cpt_for_epoch_range(epoch_start,
                                                 last_fixed_epoch)  # legacy of rolling horizon

        if not Configuration.make_rolling:
            last_fixed_epoch = epoch_end

        # DEV: this remains awaiting for another use of rolling heuristics.
        prev_separated_vars = OutputData({}, pd.DataFrame(), {}, pd.DataFrame())

        logger.info("Building first model to optimize workers.\n")

        short_term_formulation = ShortTermFormulation(
            epoch_start,
            epoch_end,
            last_fixed_epoch,
            max(d.df["handling_idx"].max() for d in self.data_holders.values()),
            prev_separated_vars,
            self.shift_params,
            self.data_holders, 
            self.params,
            self.raw_eh_parameters,
            self.previous_solution, 
            {},
            self.human_epoch)
        # TODO: this 'if' block is repeated. Make a method of it.
        if short_term_formulation:
            with cplex.Cplex() as cpx:
                short_term_formulation.set_cplex(cpx)
                short_term_formulation.define()
                short_term_formulation.declare_global_shift_wrkrs()
                short_term_formulation.declare_polys_shift_assigned()
                short_term_formulation.declare_polys_hour_assigned()
                if Configuration.activate_hourly_workers:
                    short_term_formulation.declare_hourly_workers()

                for proc in short_term_formulation.process_stages.values():
                    # declaration and setting for each of Inbound and Outbound
                    proc.declare_workers()
                    proc.declare_items()
                    proc.declare_available_wrkrs_qntty()
                    proc.declare_hired_wrkrs_qntty()
                    proc.declare_dismissed_wrkrs_qntty()

                    if Configuration.activate_transfer:
                        proc.declare_transferred_wrkrs_qntty()

                    proc.set_total_permanents_equalities()
                    proc.set_inner_constraints()
                    proc.set_reps_evolution_restrictions()
                    proc.set_available_reps_natural_constraint()

                short_term_formulation.declare_global_minmaxs()
                short_term_formulation.set_global_minmax_constraints()
                short_term_formulation.set_linking_constraints()
                short_term_formulation.set_work_capacity_constraint()
                short_term_formulation.set_polyvalent_presenteeism_constraint()
                short_term_formulation.set_real_totals_per_stage_constraint()

                if Configuration.activate_extra_hours:
                    short_term_formulation.set_extra_hours_variables_and_constraints()

                # basic_result is a model.BasicResult instance
                basic_result = short_term_formulation.run()

                if basic_result.valid:
                    self.objective_value_staffing_optimization = cpx.solution.get_objective_value();
                    logger.info("Found an optimal assignment of workers.\n")
                    self.previous_solution = basic_result.cplex_solution_list

                else:
                    logger.error(f"\nCould not solve model. Phase: "
                                 f"{BasicModel.PHASE_NAMES[BasicModel.run_number]}.\n"
                                 f"Please read {FileNames.LOG_FILE}, "
                                 f"check possible infeasibilities and/or contact support.")
                    input("Press ENTER to quit.")
                    exit()
        else:
            logger.critical(f"\nCould not build model. Phase: "
                            f"{BasicModel.PHASE_NAMES[BasicModel.run_number]}.\n"
                            f"End epoch is not greater than start epoch.\n\n Pleas contact support.")
            input("Press ENTER to quit.")
            exit()

        if DevelopDumping.DEV and DevelopDumping.MAKE_TABLE:
            con = sqlite3.connect(FileNames.DATABASE_MODEL)
            short_term_formulation.shift_params.df_workers_costs.to_sql(name='in_' + FileNames.WORKERS_COSTS[:-4],
                                                                        con=con,
                                                                        if_exists='replace',
                                                                        index=False)

            readers.auxiliary_standard_read(FileNames.WORKERS_INITIAL).to_sql(
                name='in_' + FileNames.WORKERS_INITIAL[:-4],
                con=con,
                if_exists='replace',
                index=False)

            if Configuration.activate_transfer:
                short_term_formulation.df_transfers.to_sql(name='in_' + FileNames.TRANSFERS[:-4], con=con,
                                                           if_exists='replace',
                                                           index=False)
            con.commit()
            con.close()

        if Configuration.anticipate_backlog:
            logger.info("Building second model to anticipate backlogs.\n")

            short_term_formulation = ShortTermFormulation(
                epoch_start,
                epoch_end,
                last_fixed_epoch,
                max(d.df["handling_idx"].max() for d in self.data_holders.values()),
                prev_separated_vars,
                self.shift_params,
                self.data_holders,
                self.params,
                self.raw_eh_parameters,
                basic_result.cplex_solution_list,
                basic_result.cplex_ids,
                self.human_epoch)
            if short_term_formulation:
                with cplex.Cplex() as cpx_stock:

                    short_term_formulation.set_cplex(cpx_stock)

                    short_term_formulation.define()
                    short_term_formulation.declare_polys_shift_assigned()
                    short_term_formulation.declare_polys_hour_assigned()
                    short_term_formulation.set_optimal_polyvalent_values()

                    if Configuration.activate_hourly_workers:
                        short_term_formulation.declare_hourly_workers()
                        short_term_formulation.set_optimal_hourly_constraint()

                    for proc in short_term_formulation.process_stages.values():
                        # stock anticipation declaration and setting
                        proc.declare_workers()
                        proc.declare_items()
                        proc.set_inner_constraints()
                        proc.set_optimal_permanent_reps_values()  # fix amount of perms to anticipate backlog
                        proc.declare_and_constraint_stock_variables()

                    short_term_formulation.set_work_capacity_constraint()

                    # TODO: separate the declaration of the extra_hs reps and set the 
                    #       optimal values for the second run
                    # if Configuration.activate_extra_hours:
                    #     short_term_formulation.set_extra_hours_variables_and_constraints()

                    basic_result = short_term_formulation.run()

                    if basic_result.valid:
                        self.objective_value_stock_anticipation = cpx_stock.solution.get_objective_value()
                        logger.info('Found an optimal distribution of backlogs.\n')

                    else:
                        logger.error(f"\nCould not solve model. "
                                     f"Phase: {BasicModel.PHASE_NAMES[BasicModel.run_number]}.\n"
                                     f"Please read {FileNames.LOG_FILE}, "
                                     f"check possible infeasibilities and/or contact support.")
                        input("Press ENTER to quit.")
                        exit()
            else:
                logger.critical(f"\nCould not build model. "
                                f"Phase: {BasicModel.PHASE_NAMES[BasicModel.run_number]}.\n"
                                f"End epoch is not greater than start epoch. Contact support.")
                input("Press ENTER to quit.")
                exit()

        if DevelopDumping.MAKE_OUTPUT:
            logger.info("Starting to make output (this may take a few minutes, please don't close this window).\n")

            made_output = short_term_formulation.make_output()

            RollingExecutor.write_separate_results(made_output)

            if DevelopDumping.DEV and DevelopDumping.MAKE_TABLE and not Configuration.anticipate_backlog:
                short_term_formulation.sol_to_sql()

            if DevelopDumping.DEV or DevelopDumping.QAS or Configuration.generate_validation_files:
                logger.info(f"Performing output checks. Writing results in folder {FileNames.VALIDATION_FOLDER}.\n")

                output_checker = OutputTest(self.shift_params, self.params[Process.INBOUND],
                                            self.params[Process.OUTBOUND], self.objective_value_staffing_optimization,
                                            self.objective_value_stock_anticipation)

                output_checker.generate_check_tables()

        return prev_separated_vars
