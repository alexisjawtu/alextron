import cplex
import logging
import pprint

import pandas as pd
import numpy as np

from datetime import datetime
from typing import Dict, List

import kernel.data_frames_field_names as fld_names

from kernel.data.helpers import *
from kernel.data.shift import ShiftKind, TabulatedShiftMapper
from kernel.formulation.formulation import BasicResult
from kernel.stages.short_term_model import ShortTermModel
from kernel.general_configurations import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
            shift_mapper: TabulatedShiftMapper,
            shift_holder: ShiftHolder,
            data_holders: Dict[Process, DataHolder],
            params: Dict[Process, TabulatedParameterHolder],
            previous_cplex_solution: List[float],
            previous_cplex_ids: List[int]
    ) -> None:

        self.inbound_stages: int = 0
        self.outbound_stages: int = 0

        # global minmax and workers variables: 'shift' is a number, 'shift_kind' is ShiftKind.
        self.max_wrkrs_per_shift_kind_global = {}
        self.total_wrkrs_per_shift_and_stage = {}

        self.v_polys_shift_assigned: Dict = {}  # staff that is working as polyvalent at stage j and SHIFT s
        self.v_polys_hour_assigned: Dict = {}  # TODO: Replace the following with this!
        self.wt: Dict = {}  # w_m_ft_s_t: non--specialist staff at stage j and SLOT t
        self.orig_dest: List = []  # list of tuples origin--destination to track polyvalence

        self.v_hourly_workers: Dict = {}  # Hourly workers, indexed as xtr_hr[process][stage][hour]

        self.dc_workers_initial_inbound: Dict = {}
        self.dc_workers_initial_outbound: Dict = {}

        self.start: int = start
        self.end: int = end
        self.fixed_end = fixed_end
        self.max_epoch = max_epoch  # max outbound input epoch
        self.shift_numbers = []  # Unique indices of shifts for the whole exercise

        self.cpx = None
        self.sol = []  # The list of current cplex solution values

        self.previous_cplex_solution = previous_cplex_solution  # Last cplex solution list of values
        self.previous_made_output = previous_made_output
        self.previous_cplex_ids = previous_cplex_ids

        # this is only a record of which processes where included in the input.
        self.input_processes = data_holders.keys()

        # process_stages has the two short_term_model.ShortTermModel objects, built from input.
        self.process_stages: Dict[Process, ShortTermModel] = dict()

        self.shift_mapper = shift_mapper
        self.shift_holder = shift_holder
        self.data_holders = data_holders
        self.params = params
        self.min_stage_for_cpt: Dict = {}
        self.shift_names: Dict = {}
        self.work_day_name: Dict = {}
        self.stage_names: Dict = {'INBOUND': {0: 'receiving', 1: 'check_in'},
                                  'OUTBOUND': {0: 'picking', 1: 'packing'}}
        self.human_epoch: Dict = {}

        self.df_transfers: pd.DataFrame = pd.DataFrame()
        self.dc_shifts_from: Dict = {}
        self.dc_shifts_to: Dict = {}

        self.ls_shift_kinds: List = list(ShiftKind)

        self.dc_contract_modalities_for_shift_name: Dict = {}
        self.dc_shift_names_per_contract_modality: Dict = {}
        self.dc_contract_types_and_modalities: Dict = {}

        self.polyvalence_parameters: pd.DataFrame = pd.DataFrame()
        self.absence_table: pd.DataFrame = pd.DataFrame()

        self.list_shifts_for_epoch_range = self.shift_holder.shifts_for_epoch_range(self.start, self.end)

        self.dict_stage_weights_w = {}
        self.dict_stage_weights_x = {}
        self.dict_presence_rate = {}
        self.dict_absence_rate = {}
        self.dict_max_workers_w = {}

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

        assig = assig.round({'permanents_shift': 2,
                             'permanents_hour': 2,
                             'polyvalents_shift': 2,
                             'polyvalents_hour': 2,
                             'extras_for_this_shift': 2,
                             'permanents_hour_total': 2,
                             'polyvalents_hour_total': 2,
                             'extras_hour': 2,
                             'total_shift': 2,
                             'total_hour': 2})

        return proc, assig

    def unused_rate(self, from_to: tuple, process: Process, modality: str, t: int, s: int) -> float:
        return self.params[Process(from_to[0])].presence_rate(modality, t, s) * self.params[process].absence_rate(
            modality, t, s)

    def get_epochs_scope(self, _shift_):
        # TODO: this one is a candidate to vectorization!
        epochs = self.shift_holder.epochs_for_shift[_shift_]
        return epochs[(self.start <= epochs) & (epochs <= self.end)]

    def set_human_epoch(self):
        shifts_data = self.shift_holder.shifts_df
        self.shift_names = shifts_data.groupby('shift_idx')[fld_names.SHIFT_NAME].first().to_dict()
        self.work_day_name = shifts_data.groupby(['idx', 'shift_idx'])['work_day_name'].first().to_dict()

        sort_data = shifts_data[fld_names.EPOCH_TIMESTAMP].sort_values()
        range_min = sort_data.iloc[0]
        range_max = sort_data.iloc[-1] + pd.Timedelta(hours=1)
        self.human_epoch = pd.date_range(range_min, range_max, freq='H').to_frame(
            name=fld_names.EPOCH_TIMESTAMP,
            index=False
        ).rename(columns={'index': 'idx'}).to_dict()[fld_names.EPOCH_TIMESTAMP]

    def make_output(self) -> OutputData:
        self.set_human_epoch()
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
                'hourly_workers': round(self.sol[self.v_hourly_workers[stm.process.value][j][t]],2)}
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

        for name, stm in iter(self.process_stages.items()):
            shifts = stm.sh.shifts_for_epoch_range(self.start, self.fixed_end)
            polys_follow_up = pd.concat([polys_follow_up,
                                         pd.DataFrame.from_records(
                                             [{
                                                 'shift': s,
                                                 'epoch': t,
                                                 'contract_modality': m,
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
                                                     self.sol[self.wt[m][from_to][s][t]],
                                                 'poly_hour_received_total':
                                                     self.div_or_zero(self.sol[self.wt[m][from_to][s][t]],
                                                                      self.unused_rate(from_to, name, m, t, s)),
                                                 'poly_shift': max([self.div_or_zero(
                                                     self.sol[self.wt[m][from_to][s][t]],
                                                     self.unused_rate(from_to, name, m, t_i, s))
                                                     for t_i in self.get_epochs_scope(s)])}
                                                 for s in shifts
                                                 for t in self.get_epochs_scope(s)
                                                 for m in list_modality
                                                 for from_to in self.orig_dest if
                                                 self.sol[self.wt[m][from_to][s][t]] > 0.001])])

            self.min_stage_for_cpt = stm.min_stage_for_cpt

            stages_dict_received = {stage: {from_to for from_to in self.orig_dest if
                                            (from_to[2] == stm.process.value) and (from_to[3] == stage)} for stage in
                                    range(stm.stages)}

            workers_t, workers = self.read_workers(name, stm, list_modality, stages_dict_received)

            shift_count = stm.parameters.coefficients_table.groupby('idx')[
                'shifts_for_ts'].count()
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
                for m in list_modality
                for y, shift in self.make_ys(m, j, t, shift_count[t],
                                             self.sol[stm.y[cpt][j][t]], name, workers_t, stages_dict_received[j])
            ]
            processing_pd = pd.DataFrame()

            if processing_records:
                processing_pd = pd.DataFrame.from_records(processing_records)
                processing_pd = processing_pd[processing_pd.processed > 0]

            records = []
            for m in list_modality:
                for s in shifts:
                    for t in self.get_epochs_scope(s):
                        for j in range(stm.stages):
                            wrkrs_t = workers_t[m][j][t][s]
                            wrkrs = workers[m][j][s]
                            records.append({
                                'shift': s,
                                'epoch': t,
                                'stage': j,
                                'permanents_shift': self.round_one(wrkrs['max_xt']),
                                'permanents_hour': self.round_one(wrkrs_t['xt']),
                                'permanents_hour_total': self.round_one(wrkrs_t['xt_total']),
                                'polyvalents_shift': self.round_one(wrkrs['max_wt']),
                                'polyvalents_hour': self.round_one(wrkrs_t['wt']),
                                'polyvalents_hour_total': self.round_one(wrkrs_t['wt_total']),
                                'extras_for_this_shift': 0,
                                'process': stm.process.name,
                                'extras_hour': 0,
                                'total_shift': self.round_one(wrkrs['total_perms_shift_assigned']),
                                'total_hour': self.round_one(wrkrs_t['zt']),
                                'stage_name': self.stage_names[stm.process.name][j]
                            })
            assignments_pd = pd.DataFrame.from_records(records)

            assignments_pd = assignments_pd.reindex(
                columns=['shift', 'epoch', 'stage', 'permanents_shift', 'permanents_hour',
                         'polyvalents_shift', 'polyvalents_hour', 'extras_for_this_shift',
                         'process',
                         'stage_name', 'epoch_ts', 'day', fld_names.SHIFT_NAME,
                         'permanents_hour_total',
                         'polyvalents_hour_total', 'extras_hour', 'total_shift',
                         'total_hour'], copy=False)
            processing_pd, assignments_pd = self.put_friendly_fields(processing_pd, assignments_pd)
            processing[name] = processing_pd
            assignments[name] = assignments_pd

        polys_follow_up = polys_follow_up[polys_follow_up.poly_hour_received > 0.00001]
        return OutputData(assignments, assignments_hourly, processing, polys_follow_up)

    def read_workers(self, name, stm, list_modality, stages_dict_received):
        # TODO optimize this function
        workers_t = {}
        workers = {}
        for m in list_modality:
            workers_tm = {}
            workers_t[m] = workers_tm
            workers_m = {}
            workers[m] = workers_m
            for j in range(stm.stages):
                stages_group_received = stages_dict_received[j]

                workers_tmj = {}
                workers_tm[j] = workers_tmj
                workers_mj = {}
                workers_m[j] = workers_mj
                for s in stm.sh.shifts_for_epoch_range(self.start, self.end):
                    workers_mj[s] = {
                        "total_perms_shift_assigned": abs(self.sol[stm.v_total_perms_shift_assigned[m][j][s]]),
                    }

                for t in range(stm.start, stm.end + 1):
                    workers_tmjt = {}
                    workers_tmj[t] = workers_tmjt

                    cpts = [c for c in stm.dh.dict_cpts_for_epoch[t] if self.min_stage_for_cpt[c] <= j]
                    total_processed = round(sum(self.sol[stm.y[cpt][j][t]]
                                                for cpt in cpts), 4)
                    shifts = self.shift_holder.shifts_for_epoch.get(t, [])
                    total_capacity = 0
                    for s in shifts:
                        # TODO sacar esto a un dict
                        unused_rate = stm.parameters.presence_rate(m, t, s) * stm.parameters.absence_rate(m, t, s)

                        xt = abs(self.sol[stm.xt[m][j][s][t]])
                        zt = abs(self.sol[stm.zt[m][j][s][t]])
                        weights_x = self.dict_stage_weights_x[name][m][s][j]
                        total_capacity += weights_x * xt
                        for from_to in stages_group_received:
                            total_capacity += self.dict_stage_weights_w[name][m][s][from_to] * \
                                              abs(self.sol[self.wt[m][from_to][s][t]])
                        workers_tmjt[s] = {"xt": xt,
                                           "xt_total": self.div_or_zero(xt, unused_rate),
                                           "zt": zt}

                    total_capacity = round(total_capacity, 4)

                    for s in shifts:
                        wt = 0
                        wt_total = 0
                        for from_to in stages_group_received:
                            unused_rate = self.unused_rate(from_to, stm.process, m, t, s)
                            wt += abs(self.sol[self.wt[m][from_to][s][t]])
                            wt_total += self.div_or_zero(abs(self.sol[self.wt[m][from_to][s][t]]),unused_rate)
                        workers_tmjt[s]['wt'] = wt
                        workers_tmjt[s]['wt_total'] = wt_total

                        if total_processed < total_capacity:
                            # TODO sacar esto a un dict
                            unused_rate = stm.parameters.presence_rate(m, t, s) * stm.parameters.absence_rate(m, t,
                                                                                                              s)
                            xt = self.div_or_zero(total_processed,self.dict_stage_weights_x[name][m][s][j])
                            workers_tmjt[s]['xt'] = xt
                            workers_tmjt[s]['xt_total'] = self.div_or_zero(xt, unused_rate)

                for s in stm.sh.shifts_for_epoch_range(self.start, self.fixed_end):
                    max_x = 0
                    max_w = 0
                    for t in self.get_epochs_scope(s):
                        xt = workers_tmj[t][s]['xt']
                        if xt > max_x:
                            max_x = xt

                        wt = workers_tmj[t][s]['wt']
                        if wt > max_w:
                            max_w = wt

                    workers_mj[s]['max_xt'] = max_x
                    workers_mj[s]['max_wt'] = max_w
        return workers_t, workers

    def round_one(self, number):
        if 1 > number > 0.0001:
            return np.ceil(number)
        return number

    def make_ys(self, m, j, t, shift_count, y, name, workers_t, stages_dict_received):
        shifts = self.shift_holder.shifts_for_epoch.get(t, [])
        if shift_count == 1:
            return [[y, shifts[0]]]
        work_force_total_hour = 0
        work_force_shift = {}
        for shift in shifts:
            xt = workers_t[m][j][t][shift]['xt'] * self.dict_stage_weights_x[name][m][shift][j]
            wt = sum(abs(self.sol[self.wt[m][from_to][shift][t]]) * self.dict_stage_weights_w[name][m][shift][
                        from_to] for from_to in stages_dict_received)
            aux = xt + wt
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
            stm.v_perms_shift_assigned,
            stm.xt,
            self.v_polys_shift_assigned,
            self.wt,
            stm.v_total_perms_shift_assigned,
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
