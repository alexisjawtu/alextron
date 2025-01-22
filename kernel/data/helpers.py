import numpy as np
import pandas as pd
import logging

from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List, Dict, Any, Callable

import kernel.data_frames_field_names as fld_names

from kernel.data.shift_parameters import ShiftParametersGenerator
from kernel.general_configurations import FileNames


day_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class OutputData:
    workers: dict
    hourly_workers: pd.DataFrame
    processed: dict
    polys_follow_up: pd.DataFrame


@dataclass
class CplexIds:
    """We must recover the original indices of the variables of the model from
    the optimization of the workers and pass them to the optimization of the stocks"""
    dc_x_fixed: Dict
    dc_xt_fixed: Dict
    dc_w_fixed: Dict
    dc_wt_fixed: Dict
    dc_z_fixed: Dict
    dc_zt_fixed: Dict
    dc_hourly: Dict


class Process(Enum):
    INBOUND = 0
    OUTBOUND = 1


class DataHolder:
    def __init__(
            self,
            df: pd.DataFrame,
            process: Process,
            stages: int,
            stage_names: Dict,
            max_epoch_id: int,
            initial: Dict = {}
    ) -> None:

        self.df: pd.DataFrame = df
        self.initial: Dict[tuple, int] = initial
        self.stages: int = stages
        self.stage_names = stage_names
        self.cpt_fld = DataHolder.cpt_field(process)

        self.cpts = sorted(self.df[self.cpt_fld].unique())

        min_epoch_for_cpt_dict = self.df.groupby(self.cpt_fld)['handling_idx'].min().to_dict()

        for initial_backlog_stage_cpt_pair in self.initial.keys():
            min_epoch_for_cpt_dict[initial_backlog_stage_cpt_pair[1]] = 0

        self.min_epoch_for_cpt = min_epoch_for_cpt_dict

        # demand is the very item--volumes--input: inbound.csv and outbound.csv. Goes for stage 0 only 
        self.demand = self.df.groupby(['handling_idx', self.cpt_fld])['count'].sum().to_dict()

        self.dict_cpts_for_epoch = {}

        for epoch in range(max_epoch_id + 1):
            _cpts_ = np.sort(self.df[(self.df[fld_names.EPOCH_ID] <= epoch) &
                                     (self.df[self.cpt_fld] > epoch)][self.cpt_fld].unique())
            _cpts_ = np.unique(np.hstack((_cpts_, np.array([b for a, b in self.initial.keys() if b > epoch]))))
            self.dict_cpts_for_epoch[epoch] = _cpts_

    @staticmethod
    def cpt_field(process):
        if process == Process.INBOUND:
            return 'sla_idx'
        elif process == Process.OUTBOUND:
            return 'cpt_idx'
        else:
            raise ValueError('unknown process: %s', process)

    def initial_backlogs(self, cpt, stage):
        return self.initial.get((stage, cpt), 0) if self.initial is not None else 0

    def inserted_demand(self, epoch):
        return int(self.df[self.df['handling_idx'] == epoch]['count'].sum())

    def cpts_for_epoch_range(self, start, end):
        """ if start is less than a cpt from the initial
            backlogs, we must include that cpt because those are independent tables. """
        data = self.df[(self.df['handling_idx'] <= end) &
                       (self.df[self.cpt_fld] > start)][self.cpt_fld].sort_values().unique()

        data = np.unique(np.hstack((data, np.array([b for a, b in self.initial.keys() if b > start]))))
        return [int(c) for c in data]

    def min_stages_for_cpts(self) -> Dict[int, int]:
        # TODO: calculate only once! Now we calculate this for every interval rolled
        v = np.unique(self.df[self.cpt_fld])
        min_dict = dict(zip(v, [0 for i in v]))
        for j, c in self.initial:
            if c not in min_dict:
                min_dict[c] = j
            else:
                min_dict[c] = min(min_dict[c], j)
        min_dict = {c: int(min_dict[c]) for c in sorted(min_dict.keys())}
        return min_dict


class TabulatedParameterHolder:
    def __init__(
            self,
            process_name: str,
            shift_params: ShiftParametersGenerator,
            backlog_bounds: pd.DataFrame,
            process_workers_full_table: pd.DataFrame,
            polyvalents_parameters: pd.DataFrame,
            raw_shifts: pd.DataFrame,
            presence_coefs_raw: pd.DataFrame,
            presence_coefs_sched: pd.DataFrame,
            absence_table: pd.DataFrame,
            absence_table_scheduling: pd.DataFrame,
            hourly_workers_cost: float,
            hourly_work_force: float
    ) -> None:

        self.process_name = process_name
        self.shift_params = shift_params

        self.coefficients_table = shift_params.shifts_df.copy()

        self.fractions = dict()

        self.backlog_bounds = backlog_bounds

        self.process_workers_full_table = process_workers_full_table
        self.polyvalents_parameters = shift_params.polyvalents_parameters
        self.raw_shifts = raw_shifts
        self.presence_coefs_raw = presence_coefs_raw
        self.presence_coefs_sched = presence_coefs_sched

        self.presence_rate: Callable = None

        self.absence_table = absence_table
        self.absence_table_scheduling = absence_table_scheduling

        self.get_absence_rate_aux: Callable = None
        self.get_presence_ratio_aux: Callable = None

        self.hourly_workers_cost: float = hourly_workers_cost
        self.hourly_work_force: float = hourly_work_force

        self.presence_coefs = [],
        self.back_upper_bounds = {}
        self.back_lower_bounds = {}

        self.switch_getter_for_absence_rates()
        self.switch_getter_for_presence_ratios()
        self.set_backlog_bounds()
        self.make_ratios_for_fractional_hours()

    def make_ratios_for_fractional_hours(self) -> None:
        """
        Here we construct the numbers to use the criterion according to which

            handling == 07:24  ==>  q := 0.4

        to handle the case of hours with minutes.

        Here we return a dict like this:

            { (epoch_id, shift_id) : q }

        that is

            { (t, s) : q }

        """
        sh = self.coefficients_table[["shift_idx", "start_minute", "end_minute", "idx"]].copy()
        sh["aux_idx"] = sh["idx"]

        sh = sh.groupby(["shift_idx", "start_minute", "end_minute"])[["aux_idx", "idx"]].agg(
            {"aux_idx": min, "idx": max}).reset_index()

        # sh[fld_names.RATIO_BEFORE_STD_HOURS] = 1 - sh["start_minute"]/60
        # sh[fld_names.RATIO_AFTER_STD_HOURS] = sh["end_minute"]/60

        # self.fractions = dict(zip(zip(sh["aux_idx"], sh["shift_idx"]), sh[fld_names.RATIO_BEFORE_STD_HOURS]))
        # self.fractions.update(dict(zip(zip(sh["idx"], sh["shift_idx"]), sh[fld_names.RATIO_AFTER_STD_HOURS])))

    def switch_getter_for_absence_rates(self) -> None:
        if self.absence_table_scheduling.empty:
            self.get_absence_rate_aux = self.get_absence_rate_aux_one_table
        else:
            self.get_absence_rate_aux = self.get_absence_rate_aux_two_tables

    def switch_getter_for_presence_ratios(self) -> None:
        if self.presence_coefs_sched.empty:
            self.presence_rate = self.get_presence_ratio_aux_one_table
        else:
            self.presence_rate = self.get_presence_ratio_aux_two_tables

    def set_backlog_bounds(self) -> None:
        u_bound = float(
            self.backlog_bounds[self.backlog_bounds['process'] == self.process_name]['upper_bound'].iloc[0]
        )

        l_bound = float(
            self.backlog_bounds[self.backlog_bounds['process'] == self.process_name]['lower_bound'].iloc[0]
        )

        self.back_upper_bounds = {1: {s: u_bound for s in self.shift_params.shifts}}
        self.back_lower_bounds = {0: {s: l_bound for s in self.shift_params.shifts}}

    def max_workers_x(self, m: str, j: int, s: int) -> float:
        qry = f"stage == {j} and shift_idx == {s} and {fld_names.HIRING_MODALITY} == '{m}'"
        try:
            coef = float(self.process_workers_full_table.query(qry)["max_workers"].iloc[0])
        except IndexError:
            coef = 10
            logger.warning("No max_workers value set for %s at stage %d and epoch %d in file %s. "
                           "We will assume max_workers equal to ten." % (m, t, s, FileNames.WORKERS_PARAMETERS))
        return coef

    def max_workers_w(self, m: str, from_to: Tuple, s: int) -> float:
        # proces and stage origin and destination 
        pr_orig, stg_orig, pr_dest, stg_dest = from_to

        qry = f"process_origin == '{Process(pr_orig).name.lower()}' and " + \
              f"stage_origin == {stg_orig} and " + \
              f"process_destination == '{Process(pr_dest).name.lower()}' and " + \
              f"stage_destination == {stg_dest} and " + \
              f"{fld_names.HIRING_MODALITY} == '{m}' and " + \
              f"{fld_names.SHIFT_NAME} == '{self.shift_params.get_shift_name(s)}'"
        try:
            coef = float(self.polyvalents_parameters.query(qry)["max_workers"].iloc[0])
        except IndexError:
            coef = 10.
            logger.warning("No maximum of polyvalents set for modality %s,\n"
                           "                         movement path "
                           "(%d, %d) --> (%d, %d) for shift %d in file %s.\n"
                           "                         We will assume maximum of polyvalents equal to ten." %
                           (m, from_to[0], from_to[1], from_to[2], from_to[3], s, FileNames.POLYV_PARAMS))
        return coef

    def get_cost_hourly_workers(self) -> float:
        return self.hourly_workers_cost

    def stage_weights_x(self, m: str, j: int, s: int) -> float:
        qry = f"stage == {j} and shift_idx == {s} and {fld_names.HIRING_MODALITY} == '{m}'"
        return float(self.process_workers_full_table.query(qry)["work_force"].iloc[0])

    def stage_weights_w(self, m: str, from_to: Tuple, s: int) -> float:
        # proces and stage origin and destination 
        pr_orig, stg_orig, pr_dest, stg_dest = from_to

        qry = f"process_origin == '{Process(pr_orig).name.lower()}' and " + \
              f"stage_origin == {stg_orig} and " + \
              f"process_destination == '{Process(pr_dest).name.lower()}' and " + \
              f"stage_destination == {stg_dest} and " + \
              f"{fld_names.HIRING_MODALITY} == '{m}' and " + \
              f"{fld_names.SHIFT_NAME} == '{self.shift_params.get_shift_name(s)}'"
        try:
            coef = float(self.polyvalents_parameters.query(qry)["work_force"].iloc[0])
        except IndexError:
            logger.warning("Missing work_force of polyvalents for modality %s,\n"
                           "                         movement path "
                           "(%d, %d) --> (%d, %d) for shift %d in file %s.\n"
                           "                         We will assume this work_force equal to 20." %
                           (m, from_to[0], from_to[1], from_to[2], from_to[3], s, FileNames.POLYV_PARAMS))
            coef = 20.

        return coef

    def backlogs_upper_bounds(self, j: int, s: int) -> int:
        return self.back_upper_bounds.get(j, {}).get(s, 0)

    def backlogs_lower_bounds(self, j: int, s: int) -> int:
        return self.back_lower_bounds.get(j, {}).get(s, 0)

    def get_presence_ratio_aux_one_table(self, m: str, t: int, s: int) -> float:
        # This method is for the case of only presence_table included in input.
        ts = self.shift_params.shifts_df.query(f"idx == {t}")[fld_names.EPOCH_TIMESTAMP].iloc[0]
        sh_name = self.shift_params.get_shift_name(s)

        truth_mask = ((self.presence_coefs_raw["shift_name"] == sh_name) &
                      (self.presence_coefs_raw[fld_names.HIRING_MODALITY] == m) &
                      (self.presence_coefs_raw["hour"] == ts.hour))
        day_name = day_names[ts.dayofweek]

        try:
            coef = self.presence_coefs_raw[truth_mask][day_name].iloc[0] \
                    * self.fractions.get((t, s), 1)

        except IndexError:
            logger.warning("Missing presence ratio for %s, %s at epoch %d and shift %d in input files.\n"
                           "                         We will assume presence ratio equal to one.\n" % 
                           (self.process_name, m, t, s))
            coef = 1

        return float(coef)

    def get_presence_ratio_aux_two_tables(self, m: str, t: int, s: int) -> float:
        # This method is for the case of both presence_table and presence_table_scheduling
        # included in input.
        ts = self.shift_params.shifts_df.query(f"idx == {t}")[fld_names.EPOCH_TIMESTAMP].iloc[0]
        sh_name = self.shift_params.get_shift_name(s)
        day_name = day_names[ts.dayofweek]
        
        is_scheduled = ((self.presence_coefs_sched["hour"] == ts) &
                        (self.presence_coefs_sched["shift_name"] == sh_name) &
                        (self.presence_coefs_sched[fld_names.HIRING_MODALITY] == m))

        try:
            _fraction_ = self.fractions.get((t, s), 1)

            if is_scheduled.any():  # if this hour--shift is scheduled, read from the scheduled table.
                coef = self.presence_coefs_sched[is_scheduled]["rate"].iloc[0] \
                        * _fraction_
            
            else:
                truth_mask = ((self.presence_coefs_raw["shift_name"] == sh_name) &
                              (self.presence_coefs_raw[fld_names.HIRING_MODALITY] == m) &
                              (self.presence_coefs_raw["hour"] == ts.hour))

                coef = self.presence_coefs_raw[truth_mask][day_name].iloc[0] \
                        * _fraction_

        except IndexError:
            logger.warning("Missing presence ratio for %s, %s at epoch %d and shift %d in input files.\n"
                           "                         We will assume presence ratio equal to one.\n" % 
                           (self.process_name, m, t, s))
            coef = 1

        return float(coef)

    def get_absence_rate_aux_one_table(self, m: str, t: int, s: int, criterion_field: str) -> float:
        # This method is for the case of only absence_table included in input.
        #
        # criterion_field: justified or unjustified

        ts = self.shift_params.shifts_df.query(f"idx == {t}")[fld_names.EPOCH_TIMESTAMP].iloc[0]
        day_name = day_names[ts.dayofweek]
        sh_name = self.shift_params.get_shift_name(s)

        mask = ((self.absence_table[fld_names.HIRING_MODALITY] == m) &
                (self.absence_table["shift_name"] == sh_name) &
                (self.absence_table["day_name"] == day_name))

        try:
            coef = self.absence_table[mask][criterion_field].iloc[0]
        except IndexError:
            logger.warning("No %s value set for %s, %s at epoch %d and shift %d in file %s.\n"
                           "                         We will assume that absence ratio equal to 0.1." %
                           (criterion_field, self.process_name, m, t, s, FileNames.ABSENT_TABLE))
            coef = 0.1
        return float(coef)

    def get_absence_rate_aux_two_tables(self, m: str, t: int, s: int, criterion_field: str) -> float:
        # This method is for the case of both absence_table and absence_table_scheduling
        # included in input.
        #
        # criterion_field: justified or unjustified
        
        ts = self.shift_params.shifts_df.query(f"idx == {t}")[fld_names.EPOCH_TIMESTAMP].iloc[0]
        sh_name = self.shift_params.get_shift_name(s)
        day_name = day_names[ts.dayofweek]

        is_scheduled = ((self.absence_table_scheduling[fld_names.VAL_START] <= ts) &
                        (ts <= self.absence_table_scheduling[fld_names.VAL_END]) &
                        (self.absence_table_scheduling[fld_names.SHIFT_NAME] == sh_name) &
                        (self.absence_table_scheduling[fld_names.HIRING_MODALITY] == m))

        try:
            if is_scheduled.any():
                coef = self.absence_table_scheduling[is_scheduled][criterion_field].iloc[0]

            else:
                mask = ((self.absence_table[fld_names.HIRING_MODALITY] == m) &
                        (self.absence_table["shift_name"] == sh_name) &
                        (self.absence_table["day_name"] == day_name))

                coef = self.absence_table[mask][criterion_field].iloc[0]

        except IndexError:
            logger.warning("Missing %s value for %s, %s at epoch %d and shift %d in file %s\n"
                           "                         or in file %s.\n"
                           "                         We will assume that absence ratio equal to 0.1." %
                           (criterion_field, self.process_name, m, t, s,
                            FileNames.ABSENT_TABLE, FileNames.ABSENT_TABLE_SCHEDULING))
            coef = 0.1

        return float(coef)

    def unjustified_absence_rate(self, m: str, t: int, s: int) -> float:
        return self.get_absence_rate_aux(m, t, s, fld_names.UNJUSTIF_ABS_RATE)

    def justified_absence_rate(self, m: str, t: int, s: int) -> float:
        # this goes into the f_obj
        return self.get_absence_rate_aux(m, t, s, fld_names.JUSTIF_ABS_RATE)

    def absence_rate(self, m: str, t: int, s: int) -> float:
        # this goes into the work capacity
        return 1 - (self.unjustified_absence_rate(m, t, s) + self.justified_absence_rate(m, t, s))


class TabulatedInboundDataHolder(DataHolder):
    def __init__(self, df, process, stages, stage_names, max_epoch_id, initial=None):
        DataHolder.__init__(self, df, process, stages, stage_names, max_epoch_id, initial)


class TabulatedOutboundDataHolder(DataHolder):
    def __init__(self, df, process, stages, stage_names, max_epoch_id, initial=None, subcarriers_initial=None):
        DataHolder.__init__(self, df, process, stages, stage_names, max_epoch_id, initial)

        self.subcarriers_initial = subcarriers_initial

    def subcarriers_for_cpt_and_epoch(self, cpt, epoch):
        # TODO: should we inspect also the initial_backlog table looking for carriers?
        return np.unique(self.df[(self.df['cpt_idx'] == cpt) & (self.df['handling_idx'] == epoch)]['carrier'])

    def subcarrier_demand(self, subcarrier, cpt, slot):
        return int(self.df[(self.df['carrier'] == subcarrier) & (self.df['cpt_idx'] == cpt)
                           & (self.df['handling_idx'] == slot)]['count'])

    def initial_subcarrier_backlog(self, subcarrier, cpt, stage):
        return self.subcarriers_initial.get((stage, cpt), 0) if self.subcarriers_initial is not None else 0


@dataclass
class ExtraHoursParameters:
    """

    Simple holder of global parameters collection.

    This is perhaps a good template to design the future global ParameterHolder,
    not depending on Inbound/Outbound.

    Example of df_extra_hours_expanded_table:

    shift_idx  modality rate_of_extra_hours_acceptance max_weekly_extra_hours unitary_cost_extra_hours max_daily_extra_hours                       full_range extra_hours_ratios
           13 MELI_PERM                           3.00                     12                      1.5                     4 [10, 11, 12, 13, 25, 26, 27, 28]
           13  DHL_Temp                           1.00                     26                      2.9                     3         [11, 12, 13, 25, 26, 27]
           26 MELI_PERM                           2.86                     13                      1.6                     2                 [36, 37, 49, 50]
           26  DHL_Temp                           0.90                     27                      3.0                     4 [34, 35, 36, 37, 49, 50, 51, 52]
           32 MELI_PERM                           1.74                     21                      2.4                     3         [45, 46, 47, 58, 59, 60]
           41 MELI_PERM                           2.72                     14                      1.7                     2                 [60, 61, 73, 74]
           41  DHL_Temp                           0.76                     28                      3.1                     4 [58, 59, 60, 61, 73, 74, 75, 76]
           46 MELI_PERM                           1.60                     22                      2.5                     3         [69, 70, 71, 82, 83, 84]

    full_range: the list of indices of the extra hour epochs for that shift.
    For example, shift 13 consists exactly of epochs [14, 15, 16, ... , 24]

    extra_hours_ratios:
    ------------------

    The idea is that it works like this. For a standard shift {9:15 .. 13:20}, we have

    Missing ratios      Prev EH                     Original shift    Post EH
    (0.25, 0.67)   ---> [ 1 - 0.25, 1, 1,   0.25] + {9:15 .. 13:20} + [       0.67,  1,  1, 1 - 0.67]
                         6:15-6:59, 7, 8, 9-9:14                       13:21-13:59, 14, 15, 16-16:20

    df_extra_hours_expanded_table: pd.DataFrame
    """

    dc_acceptance_ratios: Dict  # Depends on shift_id and modality.
    dc_daily_maxs: Dict  # Depends on shift_id and modality.
    dc_presence_ratios_for_extra_hours: Dict   # TAL VEZ NO VENGA ACA
    dc_extra_hours_ratios: Dict