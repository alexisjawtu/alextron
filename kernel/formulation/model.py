import cplex
import logging

import pandas as pd
import numpy as np
import pprint

from dataclasses import dataclass
from typing import Tuple, Dict, List

import sot_fbm_staffing.data_frames_field_names as fld_names

from sot_fbm_staffing.data.helpers import *
from sot_fbm_staffing.data.shift_parameters import ShiftParametersGenerator
from sot_fbm_staffing.stages.short_term_model import ShortTermModel
from sot_fbm_staffing.general_configurations import *
from sot_fbm_staffing.util.readers import imported_shift_names_and_indices


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class BasicResult:
    valid: bool
    cplex_solution_list: List
    cplex_ids: List


class BasicModel:
    # TODO: document list of codes
    # optimal, optimal tol, node lim feas, time lim feas and 127
    ACCEPTABLE_CPLEX_CODES = {101, 102, 105, 107, 127}
    MIN_GAP = 1
    LP_NAMES = ('staffing_optimization.lp', 'stock_anticipation.lp')
    PHASE_NAMES = ('Staffing Optimization', 'Backlog Anticipation')

    # The following static counter tells us in which moment of the optimization are we
    run_number = -1

    def __init__(
            self,
            start: int,
            end: int,
            fixed_end: int,
            max_epoch: int,
            previous_made_output: Dict[Process, pd.DataFrame],
            shift_params: ShiftParametersGenerator,
            data_holders: Dict[Process, DataHolder],
            params: Dict[Process, TabulatedParameterHolder],
            raw_eh_parameters: pd.DataFrame,
            previous_cplex_solution: List[float],
            previous_cplex_ids: List[int],
            human_epoch: pd.DataFrame
    ) -> None:

        # global minmax and workers variables: 'shift' is a number
        self.max_wrkrs_per_shift_kind_global = dict()
        self.total_wrkrs_per_shift_and_stage = dict()

        self.v_polys_shift_assigned: Dict = dict()  # staff that is working as polyvalent at stage j and SHIFT s
        self.v_polys_hour_assigned: Dict = dict()  # TODO: Replace the following with this!
        self.wt: Dict = dict()  # w_m_ft_s_t: non--specialist staff at stage j and SLOT t
        self.orig_dest: List = []  # list of tuples origin--destination to track polyvalence

        self.v_extra_hours_perms: Dict = dict()
        self.v_extra_hours_polys: Dict = dict()

        self.extra_shifts_for_epoch: Dict = dict()
        self.dc_modalities_by_shift_idx_with_extra_hours: Dict = dict()
        self.dc_extra_hours_ranges: Dict = None

        self.v_hourly_workers: Dict = dict()  # Hourly workers, indexed as xtr_hr[process][stage][hour]

        self.dc_workers_initial_inbound: Dict = dict()
        self.dc_workers_initial_outbound: Dict = dict()

        self.start: int = start
        self.end: int = end
        self.fixed_end = fixed_end
        self.max_epoch = max_epoch  # max outbound input epoch
        self.shift_numbers: List = None  # Unique indices of shifts for the whole exercise

        self.cpx = None
        self.sol = []  # The list of current cplex solution values

        self.raw_eh_parameters: pd.DataFrame = raw_eh_parameters

        self.previous_cplex_solution = previous_cplex_solution  # Last cplex solution list of values
        self.previous_made_output = previous_made_output
        self.previous_cplex_ids = previous_cplex_ids

        # this is only a record of which processes where included in the input.
        self.input_processes = data_holders.keys()

        # process_stages has the two short_term_model.ShortTermModel objects, built from input.
        self.process_stages: Dict[Process, ShortTermModel] = dict()

        self.shift_params = shift_params
        self.data_holders = data_holders
        self.params = params
        self.min_stage_for_cpt: Dict = {}
        self.shift_names: Dict = {}
        self.work_day_name: Dict = {}
        self.stage_names: Dict = {'INBOUND': {0: 'receiving', 1: 'check_in'},
                                  'OUTBOUND': {0: 'picking', 1: 'packing'}}
        self.human_epoch: pd.DataFrame = human_epoch.drop(columns="idx").rename(
            columns={"date": fld_names.EPOCH_TIMESTAMP})

        self.df_transfers: pd.DataFrame = pd.DataFrame()
        self.dc_shifts_from: Dict = {}
        self.dc_shifts_to: Dict = {}

        self.dict_shift_kinds: Dict = imported_shift_names_and_indices

        self.list_modality: List = []

        self.dc_contract_modalities_for_shift_name: Dict = {}
        self.dc_shift_names_per_contract_modality: Dict = {}
        self.dc_contract_types_and_modalities: Dict = {}
        self.dc_modalities_and_contract_types: Dict = {}

        self.absence_table: pd.DataFrame = pd.DataFrame()

        self.list_shifts_for_epoch_range = self.shift_params.shifts_for_epoch_range(self.start, self.end)

        self.dict_stage_weights_w: Dict = dict()
        self.dict_stage_weights_x: Dict = dict()
        self.dict_presence_rate: Dict = dict()
        self.dict_absence_rate: Dict = dict()
        self.dict_max_workers_w: Dict = dict()

    def __bool__(self):
        return bool(self.start <= self.max_epoch)

    def put_friendly_fields(self, proc, assig):
        # output: processed
        proc['day'] = proc.apply(lambda t: self.work_day_name[(t['epoch'], t['shift'])], axis=1)

        proc[fld_names.SHIFT_NAME] = proc['shift'].apply(lambda s: self.shift_names[s])

        # output: assignment
        assig['epoch_ts'] = assig['epoch'].apply(lambda t: self.human_epoch[t])

        # TODO: PERFORMANCE try dayofweek directly in the "from_records" part
        assig[fld_names.OUTPUT_DAY] = assig.apply(lambda t: self.work_day_name[(t['epoch'], t['shift'])], axis=1)
        assig[fld_names.SHIFT_NAME] = assig['shift'].apply(lambda s: self.shift_names[s])

        proc = proc.round({'backlog': 2, 'processed': 2, 'surplus': 2})  # TODO: PLEASE BENCHMARK this!!

        assig = assig.round({"permanents_shift": 2,
                             "permanents_hour": 2,
                             "permanents_hour_fix": 2,
                             "polyvalents_shift": 2,
                             "polyvalents_hour": 2,
                             "extras_for_this_shift": 2,
                             "permanents_hour_total": 2,
                             "polyvalents_hour_total": 2,
                             "extras_hour": 2,
                             "total_shift": 2,
                             "total_hour": 2})

        return proc, assig

    def polyvalence_unused_rate(self, from_to: tuple, process: Process, modality: str, t: int, s: int) -> float:
        return self.params[Process(from_to[0])].presence_rate(modality, t, s) * self.params[process].absence_rate(
            modality, t, s)

    def get_epochs_scope(self, _shift_):
        # TODO: this one is a candidate to vectorization!
        epochs = self.shift_params.epochs_for_shift[_shift_]
        return epochs[(self.start <= epochs) & (epochs <= self.end)]

    def set_human_data(self) -> None:
        """
        This is to make our numeric solution readable by users:

        Examples:
        - shift = 17 turns some "MORNING0" shift_name.
        - epoch = 3  turns "2022-05-02 08:00:00".
        
        """

        self.shift_names: Dict = self.shift_params.shifts_df.groupby('shift_idx')[fld_names.SHIFT_NAME].first().to_dict()
        self.work_day_name: Dict = self.shift_params.shifts_df.groupby(['idx', 'shift_idx'])['work_day_name'].first().to_dict()
        self.human_epoch = self.human_epoch.to_dict()[fld_names.EPOCH_TIMESTAMP]

    def make_output(self) -> OutputData:
        """
            Meaning of columns of workers_[Process].csv:
            ===========================================

            direct form cpx: 

            - permanents_hour <--> stm.xt: stage--specialized staff at stage j, and shift s, and slot t, mod m
            - total_hour <--> stm.zt: total staff assigned at stage j, shift s, slot t, mod m
            - total_shift <--> stm.v_total_perms_shift_assigned aka "stm.z": total staff assigned at stage j, shift s

            other columns:
            
            - permanents_hour_total: permanents_hour / pres * aus (meaning, all the staff, including rests)
            - permanents_shift: max_{t in s} permanents_hour_total[t]

            - polyvalents_hour = Sum_{from_to s.t. "to == current stage"} stm.wt[m,from_to,s,t]
            - polyvalents_hour_total = polyvalents_hour / pres[stage_origin] * aus

            - polyvalents_shift: max_{t in s} polyvalents_hour_total[t]
        """
        self.set_human_data()
        list_modality = [modality for m in self.dc_contract_types_and_modalities
                         for modality in self.dc_contract_types_and_modalities[m]]

        assignments_hourly = pd.DataFrame()
        if Configuration.activate_hourly_workers:
            assignments_hourly = pd.DataFrame.from_records([{
                'epoch': t,
                'epoch_ts': self.human_epoch[t],
                'process': stm.process.name,
                'stage': j,
                'stage_name': self.stage_names[stm.process.name][j],
                'hourly_workers': round(self.sol[self.v_hourly_workers[stm.process.value][j][t]], 2)}
                for stm in self.process_stages.values()
                for t in range(stm.start, stm.end + 1)
                for j in range(stm.stages)
            ])
            assignments_hourly = assignments_hourly[assignments_hourly.hourly_workers > 0.01]
            if not (assignments_hourly.empty):
                logger.warning('The solution used hourly workers. Please check the file hourly_workers.csv')

        polys_follow_up = pd.DataFrame()
        processing = {}
        assignments = {}

        polys_follow_up = pd.concat([polys_follow_up,
                                     pd.DataFrame.from_records(
                                         [{
                                             'shift': s,
                                             'epoch': t,
                                             'stage_origin': from_to[1],
                                             'process_origin': Process(from_to[0]).name,
                                             'stage_name_origin': self.stage_names[Process(from_to[0]).name][
                                                 from_to[1]],
                                             'stage_destination': from_to[3],
                                             'process_destination': Process(from_to[2]).name,
                                             'stage_name_destination':
                                                 self.stage_names[Process(from_to[2]).name][from_to[3]],
                                             'epoch_ts': self.human_epoch[t],
                                             'shift_name': self.shift_names[s],
                                             'poly_hour_received':
                                                 round(self.sol[self.wt[from_to][s][m][t]], 2),
                                             'poly_hour_received_total':
                                                 round(self.div_or_zero(self.sol[self.wt[from_to][s][m][t]],
                                                                        self.polyvalence_unused_rate(from_to,
                                                                                                     Process(
                                                                                                         from_to[0]), m,
                                                                                                     t,
                                                                                                     s)), 2),
                                             'poly_shift': round(max([self.div_or_zero(
                                                 self.sol[self.wt[from_to][s][m][t]],
                                                 self.polyvalence_unused_rate(from_to, Process(from_to[0]), m, t_i, s))
                                                 for t_i in self.get_epochs_scope(s)]), 2),
                                             'contract_modality': m}
                                             for s in self.shift_numbers
                                             for t in self.get_epochs_scope(s)
                                             for m in self.shift_params.dc_modalities_by_shift_idx[s]
                                             for from_to in self.orig_dest if
                                             self.sol[self.wt[from_to][s][m][t]] > 0.001])])

        for name, stm in iter(self.process_stages.items()):
            shifts = stm.sh.shifts_for_epoch_range(self.start, self.fixed_end)

            self.min_stage_for_cpt = stm.min_stage_for_cpt

            stages_dict_received = {stage: {from_to for from_to in self.orig_dest if
                                            (from_to[2] == stm.process.value) and (from_to[3] == stage)} for stage in
                                    range(stm.stages)}

            workers_by_epoch, workers = self.get_workers_from_solution(name, stm, stages_dict_received)

            shift_count = stm.parameters.coefficients_table[["idx", "work_day_name", "shift_name"]
                ].drop_duplicates().groupby("idx")[["work_day_name", "shift_name"]].size()

            times = [t for t in shift_count.index if self.start <= t <= self.fixed_end]

            processing_records = [{
                'epoch': t,
                'stage': j,
                'cpt': cpt,
                'shift': shift,
                'backlog': self.sol[stm.b[cpt][j][t]],
                'processed': y,
                'surplus': self.sol[stm.b[cpt][j][t]] - self.sol[stm.y[cpt][j][t]],
                'process': stm.process.name,
                'stage_name': self.stage_names[stm.process.name][j],
                'epoch_ts': self.human_epoch[t],
                'cpt_ts': self.human_epoch[cpt]}
                for j in range(stm.stages)
                for t in times
                for cpt in [c for c in stm.dh.dict_cpts_for_epoch[t]
                            if c <= self.end and
                            j >= self.min_stage_for_cpt[c] and
                            round(self.sol[stm.y[c][j][t]], 2)]
                for y, shift in self.make_ys(j, t, shift_count[t],
                                             self.sol[stm.y[cpt][j][t]], name, workers_by_epoch,
                                             stages_dict_received[j])
            ]
            processing_pd = pd.DataFrame()

            if processing_records:
                processing_pd = pd.DataFrame.from_records(processing_records)
                processing_pd = processing_pd[processing_pd.processed > 0]

            records = []
            for s in shifts:
                for m in self.shift_params.dc_modalities_by_shift_idx[s]:
                    for t in self.get_epochs_scope(s):
                        for j in range(stm.stages):
                            wrkrs_t = workers_by_epoch[j][t][s][m]

                            wrkrs = workers[j][s][m]
                            records.append({
                                'shift': s,
                                'epoch': t,
                                'stage': j,
                                'contract_modality': m,
                                'permanents_shift': wrkrs['max_xt'],
                                'permanents_hour': wrkrs_t["xt"],
                                'polyvalents_shift': wrkrs['max_wt'],
                                'polyvalents_hour': wrkrs_t['wt'],
                                'extras_for_this_shift': 0,
                                'process': stm.process.name,
                                'stage_name': self.stage_names[stm.process.name][j],
                                'permanents_hour_total': wrkrs_t['xt_total'],
                                'polyvalents_hour_total': wrkrs_t['wt_total'],
                                'extras_hour': 0,
                                'total_shift': self.sol[stm.v_total_shift_assigned_per_modality[j][s][m]],
                                'total_hour': self.sol[stm.zt[j][s][m][t]]
                            })
            assignments_pd = pd.DataFrame.from_records(records)
            processing_pd, assignments_pd = self.put_friendly_fields(processing_pd, assignments_pd)

            assignments_pd = assignments_pd.reindex(
                columns=['shift',
                         'epoch',
                         'stage',
                         'permanents_shift',
                         'permanents_hour',
                         'polyvalents_shift',
                         'polyvalents_hour',
                         'extras_for_this_shift',
                         'process',
                         'stage_name',
                         'epoch_ts',
                         'day',
                         fld_names.SHIFT_NAME,
                         'permanents_hour_total',
                         'polyvalents_hour_total',
                         'extras_hour',
                         'total_shift',
                         'total_hour',
                         'contract_modality'])

            processing[name] = processing_pd
            assignments[name] = assignments_pd

        if not polys_follow_up.empty:
            polys_follow_up = polys_follow_up[polys_follow_up.poly_hour_received > 0.00001]

        return OutputData(assignments, assignments_hourly, processing, polys_follow_up)

    def get_workers_from_solution(self, name: Process, stm, stages_dict_received) -> Tuple[Dict]:
        # Performances:
        # 1- hacer una sola llamada de esto; muchas recorridas de (t, shifts_for_epoch) son lo mismo
        # 2- the force_shares of shift -- epoch (and similar stuff) can be tabulated in dfs and then 
        #    rapidly grouped at the beggining of the method
        # 3- tabular las listas de cpts segun epoch. review the uses of stm.dh.dict_cpts_for_epoch and make a table for each "(t, j)" at once.
        #    estamos construyendo la lista 
        #            cpts = [c for c in stm.dh.dict_cpts_for_epoch[t] if self.min_stage_for_cpt[c] <= j]
        #    cada vez
        # 4- a las epochs en las que hace falta trim, las estamos escribiendo dos veces, tal vez bifurcar entre "si trim", "no trim"?
        # 5- cosas como estas
        #                    stm.parameters.presence_rate()
        #    tabularlas previamente (como venimos haciendo)
        # 6- en el calculo de
        #       polyv_force_share_of_shift_s
        #    una vez que esté vectorizado, filtrar np.where != 0
        # 7- cambiar el uso esperado en make_ys, y en el armado de tabla

        workers_by_epoch = {}
        workers = {}
        list_modality = [modality for m in self.dc_contract_types_and_modalities
                         for modality in self.dc_contract_types_and_modalities[m]]

        for j in range(stm.stages):
            stages_group_received = stages_dict_received[j]  # ALL POSSIBLE stages of visitors: all of them except
            # for the current one.

            workers_by_epoch_j = {}
            workers_by_epoch[j] = workers_by_epoch_j
            workers_j = {}
            workers[j] = workers_j

            for s in self.shift_numbers:
                workers_js = {}
                workers_j[s] = workers_js
                for modality in stm.sh.dc_modalities_by_shift_idx[s]:
                    workers_js[modality] = {
                        "total_perms_shift_assigned": abs(
                            self.sol[stm.v_total_shift_assigned_per_modality[j][s][modality]]),
                    }

            for t in range(stm.start, stm.end + 1):
                workers_by_temp_jt = {}
                workers_jt = {}
                workers_by_epoch_j[t] = workers_by_temp_jt
                workers_j[t] = workers_jt

                cpts = [c for c in stm.dh.dict_cpts_for_epoch[t] if self.min_stage_for_cpt[c] <= j]

                # Trim rutine
                shifts: List = self.shift_params.shifts_for_epoch.get(t, [])  # Tabular si hace falta
                total_work_force = 0

                dc_unused_rates = {}  # use post rutine, only for z_trim

                dc_x_contributions = {}
                dc_weight_x = {}

                dc_wt_used = {}
                dc_wt_total = {}
                for s in shifts:
                    dc_wt_used_s = {}
                    dc_wt_total_s = {}
                    dc_unused_rates_s = {}
                    dc_x_contributions_s = {}
                    dc_weight_x_s = {}

                    dc_wt_used[s] = dc_wt_used_s
                    dc_wt_total[s] = dc_wt_total_s
                    dc_unused_rates[s] = dc_unused_rates_s
                    dc_x_contributions[s] = dc_x_contributions_s
                    dc_weight_x[s] = dc_weight_x_s

                    modalities = self.shift_params.dc_modalities_by_shift_idx[s]
                    # para mantener los indices correctos itera todas las modalidades,
                    # pero solo hace las cuentas necesarias si la modalidad corresponde al turno
                    for m in list_modality:
                        if m in modalities:
                            dc_unused_rates_s[m] = stm.parameters.presence_rate(m, t, s) * stm.parameters.absence_rate(
                                m, t, s)
                            if dc_unused_rates_s[m] == 0:
                                dc_unused_rates_s[m] = 1

                            used_wt = 0
                            total_wt = 0

                            wt_work_force = 0
                            for from_to in stages_group_received:
                                wt = abs(self.sol[self.wt[from_to][s][m][t]])
                                used_wt += wt
                                total_wt += self.div_or_zero(wt,
                                                             self.polyvalence_unused_rate(from_to, stm.process, m, t,
                                                                                          s))

                                wt_work_force += self.dict_stage_weights_w[name][s][m][from_to] * wt

                            dc_wt_used_s[m] = used_wt
                            dc_wt_total_s[m] = total_wt

                            weights_x = self.dict_stage_weights_x[name][s][m][j]
                            dc_weight_x_s[m] = weights_x
                            xt_work_force = abs(self.sol[stm.xt[j][s][m][t]]) * weights_x
                            dc_x_contributions_s[m] = xt_work_force

                            total_work_force += wt_work_force + xt_work_force
                        else:
                            dc_weight_x_s[m] = 1
                            dc_unused_rates_s[m] = 1
                            dc_x_contributions_s[m] = 0
                            dc_wt_used_s[m] = 0
                            dc_wt_total_s[m] = 0

                df_contributions = pd.DataFrame.from_dict(dc_x_contributions, orient='index',
                                                          columns=list_modality) / max(total_work_force, 1)
                df_weight_x = pd.DataFrame.from_dict(dc_weight_x, orient='index', columns=list_modality)
                df_unused_rates = pd.DataFrame.from_dict(dc_unused_rates, orient='index', columns=list_modality)
                total_processed = float(round(sum(self.sol[stm.y[cpt][j][t]] for cpt in cpts), 4))

                df_x_trim = (df_contributions * total_processed) / df_weight_x
                df_z_trim = df_x_trim / df_unused_rates
                df_used_wt = pd.DataFrame.from_dict(dc_wt_used, orient='index', columns=list_modality)
                df_total_wt = pd.DataFrame.from_dict(dc_wt_total, orient='index', columns=list_modality)

                for s in shifts:
                    workers_by_temp_jt[s] = {}
                    for m in self.shift_params.dc_modalities_by_shift_idx[s]:
                        workers_by_temp_jt[s][m] = {"xt": df_x_trim.loc[s, m],
                                                    "xt_total": df_z_trim.loc[s, m],
                                                    "wt": df_used_wt.loc[s, m],
                                                    "wt_total": df_total_wt.loc[s, m]}

            for s in stm.sh.shifts_for_epoch_range(self.start, self.fixed_end):
                workers_j[s] = {}
                for m in stm.sh.dc_modalities_by_shift_idx[s]:
                    workers_j[s][m] = {
                        "max_xt": max(workers_by_epoch_j[t][s][m]["xt_total"] for t in self.get_epochs_scope(s)),
                        "max_wt": max(workers_by_epoch_j[t][s][m]["wt_total"] for t in self.get_epochs_scope(s))
                    }

        return workers_by_epoch, workers

    def make_ys(self, j, t, shift_count, y, name, workers_by_temp, stages_dict_received):
        """
        Returns a list of paired (processed_items, shift_ids) for the final output.
        """
        shifts = self.shift_params.shifts_for_epoch.get(t, [])
        if shift_count == 1:
            return [[y, shifts[0]]]
        work_force_total_hour = 0
        work_force_shift = {}
        for shift in shifts:
            aux = 0
            for m in self.shift_params.dc_modalities_by_shift_idx[shift]:
                xt = workers_by_temp[j][t][shift][m]['xt'] * self.dict_stage_weights_x[name][shift][m][j]
                wt = sum(abs(self.sol[self.wt[from_to][shift][m][t]]) * self.dict_stage_weights_w[name][shift][m][
                    from_to] for from_to in stages_dict_received)
                aux += xt + wt
            work_force_total_hour += aux
            work_force_shift[shift] = aux

        return [[y * (self.div_or_zero(work_force_shift[shift], work_force_total_hour)), shift] for shift in
                shifts if work_force_shift[shift] > 0]

    def div_or_zero(self, num, den):
        if den:
            return num / den
        return 0

    def run(self) -> BasicResult:

        logger.info("Running model.\n")
        BasicModel.run_number += 1
        cplexlog = open(f"{InputOutputPaths.BASEDIR_OUT}/{FileNames.LOG_FILE}", 'a')

        self.cpx.set_results_stream(cplexlog)
        self.cpx.set_warning_stream(cplexlog)
        self.cpx.set_error_stream(cplexlog)
        self.cpx.set_log_stream(cplexlog)
        self.set_params()
        self.cpx.solve()

        # DEV
        if DevelopDumping.DEV:
            self.cpx.write(f"{InputOutputPaths.BASEDIR_OUT}/{BasicModel.LP_NAMES[BasicModel.run_number]}")

        cplex_ids = {p: CplexIds(
            stm.v_shift_assigned_per_modality,
            stm.xt,
            self.v_polys_shift_assigned,
            self.wt,
            stm.v_total_shift_assigned_per_modality,
            stm.zt,
            self.v_hourly_workers) for p, stm in iter(self.process_stages.items())}

        if self.cpx.solution.is_primal_feasible():
            self.sol = self.cpx.solution.get_values()

            cplexlog.write(f"\nObj: {self.cpx.solution.get_objective_value():.2f}, "
                           f"Ineqs: {self.cpx.linear_constraints.get_num()}, "
                           f"Vars: {self.cpx.variables.get_num()}\n")

            gap = self.cpx.solution.MIP.get_mip_relative_gap()
            # DEV
            if DevelopDumping.DEV:
                self.cpx.solution.write(f"{InputOutputPaths.BASEDIR_OUT}/raw_solution_{BasicModel.run_number}.xml")
                with open(f"{InputOutputPaths.BASEDIR_OUT}/" +
                          FileNames.VAR_NAMES % BasicModel.run_number, "w") as var_names:
                    var_names.write(pprint.pformat(self.cpx.variables.get_names()))

                with open(f"{InputOutputPaths.BASEDIR_OUT}/" +
                          FileNames.CONSTR_NAMES % BasicModel.run_number, "w") as constr_names:
                    constr_names.write(pprint.pformat(self.cpx.linear_constraints.get_names()))

                cplexlog.write(f"Relative Gap: {gap:.2f}\n")

            valid = (self.cpx.solution.get_status() in BasicModel.ACCEPTABLE_CPLEX_CODES) and gap <= BasicModel.MIN_GAP
            run_result = BasicResult(valid, self.sol, cplex_ids)

        else:
            run_result = BasicResult(False, [], cplex_ids)

        # TODO: make a string first and then put a unique ".write" line
        cplexlog.write("\n\n")
        cplexlog.close()

        return run_result

    def sol_to_sql(self) -> None:
        perms_shift_assigned_pd = pd.DataFrame()
        total_perms_shift_assigned_pd = pd.DataFrame()
        xt_pd = pd.DataFrame()
        zt_pd = pd.DataFrame()
        hourly_pd = pd.DataFrame()
        y_pd = pd.DataFrame()
        b_pd = pd.DataFrame()
        available_workers_quantity_pd = pd.DataFrame()
        hired_workers_quantity_pd = pd.DataFrame()
        dismissed_workers_quantity_pd = pd.DataFrame()
        transferred_wrkrs_qntty_pd = pd.DataFrame()
        total_wrkrs_per_modality_shift_stage_pd = pd.DataFrame()
        stock_pd = pd.DataFrame()
        for process_name, process in iter(self.process_stages.items()):

            total_wrkrs_per_modality_shift_stage_pd = total_wrkrs_per_modality_shift_stage_pd.append(
                pd.DataFrame.from_records([{
                    "modality": modality,
                    "shift": shift,
                    "process": process.name[0:3],
                    "stage": j,
                    "value": self.sol[self.total_wrkrs_per_shift_and_stage[shift][modality][process.value][j]]
                } for j in range(process.stages)
                    for process in self.process_stages
                    for shift in self.shift_numbers
                    for modality in self.shift_params.dc_modalities_by_shift_idx[shift]]))

            for m in self.dc_contract_types_and_modalities:
                for modality in self.dc_contract_types_and_modalities[m]:
                    for j in range(process.stages):
                        if not process.df_transfers.empty:
                            for i, r in process.df_transfers[
                                process.df_transfers[fld_names.HIRING_MODALITY] == modality].iterrows():
                                shift_name_from = r[fld_names.SHIFT_FROM]
                                shift_name_to = r[fld_names.SHIFT_TO]
                                transferred_wrkrs_qntty_pd = transferred_wrkrs_qntty_pd.append(
                                    pd.DataFrame.from_records([{
                                        "process": process.process.name[0:3],
                                        "stage": process.stage_names[j],
                                        "shift_from": shift_name_from,
                                        "shift_to": shift_name_to,
                                        "value": self.sol[
                                            process.v_transferred_workers_quantity[
                                                modality, j, shift_name_from, shift_name_to]]
                                    }])
                                )

                        for sh_name in self.dc_shift_names_per_contract_modality[modality]:
                            available_workers_quantity_pd = available_workers_quantity_pd.append(
                                pd.DataFrame.from_records([{
                                    "process": process.process.name[0:3],
                                    "modality": modality,
                                    "stage": process.stage_names[j],
                                    "shift": sh_name,
                                    "value": self.sol[process.v_hired_workers_quantity[modality][j][sh_name]]
                                }])
                            )

                            hired_workers_quantity_pd = hired_workers_quantity_pd.append(
                                pd.DataFrame.from_records([{
                                    "process": process.process.name[0:3],
                                    "modality": modality,
                                    "stage": process.stage_names[j],
                                    "shift": sh_name,
                                    "value": self.sol[process.v_available_workers_quantity[modality][j][sh_name]]
                                }])
                            )

                            dismissed_workers_quantity_pd = dismissed_workers_quantity_pd.append(
                                pd.DataFrame.from_records([{
                                    "process": process.process.name[0:3],
                                    "modality": modality,
                                    "stage": process.stage_names[j],
                                    "shift": sh_name,
                                    "value": self.sol[process.v_dismissed_workers_quantity[modality][j][sh_name]]
                                }])
                            )

                        for shift in process.sh.shifts_for_epoch_range(self.start, self.end):
                            if modality in self.shift_params.dc_modalities_by_shift_idx[shift]:
                                perms_shift_assigned_pd = perms_shift_assigned_pd.append(pd.DataFrame.from_records([{
                                    "process": process.process.name[0:3],
                                    "modality": modality,
                                    "stage": j,
                                    "shift": shift,
                                    "value": self.sol[process.v_shift_assigned_per_modality[j][shift][modality]]
                                }]))

                                total_perms_shift_assigned_pd = total_perms_shift_assigned_pd.append(
                                    pd.DataFrame.from_records([{
                                        "process": process.process.name[0:3],
                                        "modality": modality,
                                        "stage": j,
                                        "shift": shift,
                                        "value": self.sol[
                                            process.v_total_shift_assigned_per_modality[j][shift][modality]]
                                    }]))
                                epochs_scope = process.sh.epochs_for_shift[shift][
                                    process.sh.epochs_for_shift[shift] <= self.end]

                                for epoch in epochs_scope:
                                    xt_pd = xt_pd.append(
                                        pd.DataFrame.from_records([{
                                            "process": process.process.name[0:3],
                                            "modality": modality,
                                            "stage": j,
                                            "shift": shift,
                                            "epoch": epoch,
                                            "value": self.sol[process.xt[j][shift][modality][epoch]]
                                        }]))
                                    zt_pd = zt_pd.append(
                                        pd.DataFrame.from_records([{
                                            "process": process.process.name[0:3],
                                            "modality": modality,
                                            "stage": j,
                                            "shift": shift,
                                            "epoch": epoch,
                                            "value": self.sol[process.zt[j][shift][modality][epoch]]
                                        }]))

            cpts = [cpt for cpt in process.dh.cpts_for_epoch_range(self.start, self.end) if cpt <= self.end]
            for cpt in cpts:
                for j in range(process.min_stage_for_cpt[cpt], process.stages):
                    start_epoch = max(process.dh.min_epoch_for_cpt[cpt], self.start - 1, 0)
                    end_epoch = min(cpt - 1, self.end)
                    for epoch in range(start_epoch, end_epoch + 1):
                        if bool(process.stock):
                            stock_pd = stock_pd.append(pd.DataFrame.from_records([{
                                "process": process.process.name[0:3],
                                "cpt": cpt,
                                "stage": j,
                                "epoch": epoch,
                                "value": self.sol[process.stock[cpt][j][epoch]]
                            }]))
                        y_pd = y_pd.append(pd.DataFrame.from_records([{
                            "process": process.process.name[0:3],
                            "cpt": cpt,
                            "stage": j,
                            "epoch": epoch,
                            "value": self.sol[process.y[cpt][j][epoch]]
                        }]))
                        b_pd = b_pd.append(pd.DataFrame.from_records([{
                            "process": process.process.name[0:3],
                            "cpt": cpt,
                            "stage": j,
                            "epoch": epoch,
                            "value": self.sol[process.b[cpt][j][epoch]]
                        }]))

            if Configuration.activate_hourly_workers:
                hourly_pd = pd.DataFrame.from_records([{
                    "process": stm.process.name[0:3],
                    "stage": j,
                    "epoch": t,
                    "value": self.sol[self.v_hourly_workers[process.process.value][j][t]]
                } for stm in self.process_stages.values() for j in range(stm.stages)
                    for t in range(stm.start, stm.end + 1)])

        polys_shift_assigned_pd = pd.DataFrame.from_records([{
            "modality": modality,
            "from_to": str(from_to),
            "shift": shift,
            "value": self.sol[self.v_polys_shift_assigned[from_to][shift][modality]]
        } for shift in self.shift_numbers
            for from_to in self.orig_dest
            for modality in self.shift_params.dc_modalities_by_shift_idx[shift]])

        wd_pd = pd.DataFrame.from_records([{
            "modality": m,
            "shift": s,
            "from_to": str(from_to),
            "epoch": t,
            "value": self.sol[self.wt[from_to][s][m][t]]
        } for s in self.shift_numbers for from_to in self.orig_dest
            for t in self.shift_params.epochs_for_shift[s][
                self.shift_params.epochs_for_shift[s] <= self.end]
            for m in self.shift_params.dc_modalities_by_shift_idx[s]])

        con = sqlite3.connect(FileNames.DATABASE_MODEL)
        perms_shift_assigned_pd.to_sql(name='var_perms_shift_assigned', con=con, if_exists='replace', index=False)
        total_perms_shift_assigned_pd.to_sql(name='var_total_perms_shift_assigned', con=con, if_exists='replace',
                                             index=False)
        xt_pd.to_sql(name='var_xt', con=con, if_exists='replace', index=False)
        zt_pd.to_sql(name='var_zt', con=con, if_exists='replace', index=False)
        y_pd.to_sql(name='var_y', con=con, if_exists='replace', index=False)
        b_pd.to_sql(name='var_b', con=con, if_exists='replace', index=False)
        available_workers_quantity_pd.to_sql(name='var_available_workers_quantity', con=con, if_exists='replace',
                                             index=False)
        hired_workers_quantity_pd.to_sql(name='var_hired_workers_quantity', con=con, if_exists='replace', index=False)
        dismissed_workers_quantity_pd.to_sql(name='var_dismissed_workers_quantity', con=con, if_exists='replace',
                                             index=False)
        if not transferred_wrkrs_qntty_pd.empty:
            transferred_wrkrs_qntty_pd.to_sql(name='var_transferred_wrkrs_qntty', con=con, if_exists='replace',
                                              index=False)
        if not stock_pd.empty:
            stock_pd.to_sql(name='var_stock', con=con, if_exists='replace', index=False)
        polys_shift_assigned_pd.to_sql(name='var_polys_shift_assigned', con=con, if_exists='replace', index=False)
        wd_pd.to_sql(name='var_wd', con=con, if_exists='replace', index=False)
        total_wrkrs_per_modality_shift_stage_pd.to_sql(name='var_total_wrkrs_per_modality_shift_stage', con=con,
                                                       if_exists='replace', index=False)

        if Configuration.activate_hourly_workers & hourly_pd.empty:
            hourly_pd.to_sql(name='var_hourly', con=con, if_exists='replace', index=False)

        con.commit()
        con.close()
