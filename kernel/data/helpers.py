import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, List, Dict, Any

import kernel.data_frames_field_names as fld_names

from kernel.data.shift import TabulatedShiftMapper, Shift, ShiftKind

day_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}


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


class ShiftHolder:
    def __init__(self, df: pd.DataFrame, mapper: TabulatedShiftMapper) -> None:
        self.mapper = mapper

        self.shifts_df = df.sort_values('handling_ts')

        self.shifts_df['shift_idx'] = self.shifts_df.groupby(['calendar_reference_day', 'kind_value']).ngroup()

        # with the generality of shifts achieved, the shift with greatest index is not 
        # neccessarily the shift that ends last
        # TODO hacer un miembro aca que sea shift_that_ends_the_last =

        self.shifts = self.shifts_df['shift_idx'].unique()

        # TODO: guarantee to sort the following aggregated list
        self.shifts_for_epoch = self.shifts_df.groupby('idx')['shift_idx'].agg(list).to_dict()
        # TODO: guarantee to sort the following aggregated list
        self.epochs_for_shift = self.shifts_df.sort_values(['idx']).groupby('shift_idx')['idx'] \
            .unique().to_dict()

        self.df_workers_costs: pd.DataFrame = pd.DataFrame()


    def get_shifts(self, s: int) -> Shift:
        # This requires a shift number and returns it's Shift instance.
        return self.shifts_df.query("shift_idx == @s")["shifts_for_ts"].iloc[0]

    def get_shift_name(self, s: int) -> str:
        return self.get_shifts(s).kind.name

    def shifts_for_epoch_range(self, start: int, end: int) -> List:
        data = self.shifts_df.query('@start <= idx <= @end and shift_idx >= 0')['shift_idx'].sort_values().unique()
        return [int(s) for s in data]

    def get_cost(self, kind: ShiftKind, cost_name: str) -> float:
        """
        READ THIS:

        We put this getter here because we need the costs to be accessible in 
        both ShorTermFormulation and in ShorTermModel and there are not 
        many better options due to an over-designed architechture.
        For the same reason we have df_workers_costs.

        cost_name is HIRING_COST, DISMISSAL_COST, and so on.

        """

        # The following query is split in several lines to serve as an example of how
        # the intermediate steps work in a DataFrame object.
        bool_mask = self.df_workers_costs[fld_names.SHIFT_NAME] == kind.name

        if bool_mask.any():
            one_row_df = self.df_workers_costs[bool_mask]

            series = one_row_df[cost_name]  # requesting a column gives a Series, not a DataFrame.

            cost = series.loc[one_row_df.index.min()]  # the first index is the only index.

        else:
            logger.warning("No specified cost for %s, we will use the value 1 and the instance will keep running. "
                           "Maybe you forgot to put it in file %s.\n", kind.name, FileNames.WORKERS_COSTS)
            cost = 1

        return float(cost)


class IntervalShiftHolder(ShiftHolder):
    def __init__(
            self,
            start: pd.Timestamp,
            end: pd.Timestamp,
            interval: pd.Timedelta,
            mapper: TabulatedShiftMapper,
            index_handling_bijection: pd.DataFrame
    ) -> None:
        # TODO: all the ShiftHolder thing maybe goes within the ShiftMapper and
        # put an inclusive name

        # TODO: this algorithm goes in class TabulatedShiftMapper
        # and we will construct the whole table within de mapper
        records = []
        ts = start
        idx = 0

        while ts <= end:
            s = mapper.map_shifts(ts)
            # This is the df for ShiftHolder.shifts_df
            records += [{'work_day_name': c,
                         'handling_ts': ts,
                         'idx': idx,
                         'shifts_for_ts': a,
                         'calendar_reference_day': pd.Timestamp(ts.year, ts.month, ts.day) - pd.Timedelta(days=b),
                         'kind_value': a.kind.value,
                         'shift_name': a.kind.name,
                         'is_scheduled': 0}
                        for a, b, c in s]

            idx += 1
            ts += interval

        df = pd.DataFrame.from_records(records)
        df = mapper.map_scheduled_shifts(df, index_handling_bijection)

        ShiftHolder.__init__(self, df, mapper)


class DataHolder:
    def __init__(
            self,
            df: pd.DataFrame,
            process: Process,
            stages: int,
            max_epoch_id: int,
            initial: Dict = {}
    ) -> None:

        self.df: pd.DataFrame = df
        self.initial: Dict[tuple, int] = initial
        self.stages: int = stages

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
            shift_holder: ShiftHolder,
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
        self.shift_holder = shift_holder
        self.coefficients_table = shift_holder.shifts_df.copy()
        self.backlog_bounds = backlog_bounds

        self.process_workers_full_table = process_workers_full_table
        self.polyvalents_parameters = polyvalents_parameters

        self.raw_shifts = raw_shifts
        self.presence_coefs_raw = presence_coefs_raw
        self.presence_coefs_sched = presence_coefs_sched

        self.absence_table = absence_table
        self.absence_table_scheduling = absence_table_scheduling

        self.get_absence_rate_aux = None

        self.hourly_workers_cost: float = hourly_workers_cost
        self.hourly_work_force: float = hourly_work_force

        self.presence_coefs = [],
        self.back_upper_bounds = {}
        self.back_lower_bounds = {}

        self.switch_getter_for_absence_rates()
        self.set_backlog_bounds()

    def switch_getter_for_absence_rates(self) -> None:
        if self.absence_table_scheduling.empty:
            self.get_absence_rate_aux = self.get_absence_rate_aux_one_table
        else:
            self.get_absence_rate_aux = self.get_absence_rate_aux_two_tables

    def set_backlog_bounds(self) -> None:
        u_bound = float(
            self.backlog_bounds[self.backlog_bounds['process'] == self.process_name]['upper_bound'].iloc[0]
        )

        l_bound = float(
            self.backlog_bounds[self.backlog_bounds['process'] == self.process_name]['lower_bound'].iloc[0]
        )

        self.back_upper_bounds = {1: {s: u_bound for s in self.shift_holder.shifts}}
        self.back_lower_bounds = {0: {s: l_bound for s in self.shift_holder.shifts}}

    def max_workers_x(self, m: str, j: int, s: int) -> float:
        qry = f"stage == {j} and shift_idx == {s} and {fld_names.HIRING_MODALITY} == '{m}'"
        return float(self.process_workers_full_table.query(qry)["max_workers"].iloc[0])

    def max_workers_w(self, m: str, from_to: Tuple, s: int) -> float:
        # proces and stage origin and destination 
        pr_orig, stg_orig, pr_dest, stg_dest = from_to

        qry = f"process_origin == '{Process(pr_orig).name.lower()}' and " + \
              f"stage_origin == {stg_orig} and " + \
              f"process_destination == '{Process(pr_dest).name.lower()}' and " + \
              f"stage_destination == {stg_dest} and " + \
              f"{fld_names.HIRING_MODALITY} == '{m}' and " + \
              f"{fld_names.SHIFT_NAME} == '{self.shift_holder.get_shifts(s).kind.name}'"

        return float(self.polyvalents_parameters.query(qry)["max_workers"].iloc[0])

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
              f"{fld_names.SHIFT_NAME} == '{self.shift_holder.get_shifts(s).kind.name}'"

        return float(self.polyvalents_parameters.query(qry)["work_force"].iloc[0])

    def backlogs_upper_bounds(self, j: int, s: int) -> int:
        return self.back_upper_bounds.get(j, {}).get(s, 0)

    def backlogs_lower_bounds(self, j: int, s: int) -> int:
        return self.back_lower_bounds.get(j, {}).get(s, 0)

    def presence_rate(self, m: str, t: int, s: int) -> float:
        ts = self.shift_holder.shifts_df.query(f"idx == {t}")["handling_ts"].iloc[0]
        sh_name = self.shift_holder.get_shift_name(s)

        truth_mask = ((self.presence_coefs_raw["shift_name"] == sh_name) &
                      (self.presence_coefs_raw[fld_names.HIRING_MODALITY] == m) &
                      (self.presence_coefs_raw["hour"] == ts.hour))
        day_name = day_names[ts.dayofweek]

        coef = self.presence_coefs_raw[truth_mask][day_name].iloc[0]

        if not self.presence_coefs_sched.empty:
            is_scheduled = ((self.presence_coefs_sched["hour"] == ts) &
                            (self.presence_coefs_sched["shift_name"] == sh_name) &
                            (self.presence_coefs_sched[fld_names.HIRING_MODALITY] == m))

            if is_scheduled.any():  # if this hour--shift is scheduled, read from the scheduled table.
                coef = self.presence_coefs_sched[is_scheduled]["rate"].iloc[0]

        return float(coef)

    def get_absence_rate_aux_one_table(self, m: str, t: int, s: int, criterion_field: str):
        ts = self.shift_holder.shifts_df.query(f"idx == {t}")[fld_names.EPOCH_TIMESTAMP].iloc[0]
        day_name = day_names[ts.dayofweek]
        sh_name = self.shift_holder.get_shift_name(s)

        mask = ((self.absence_table[fld_names.HIRING_MODALITY] == m) &
                (self.absence_table["shift_name"] == sh_name) &
                (self.absence_table["day_name"] == day_name))

        coef = self.absence_table[mask][criterion_field].iloc[0]

        return float(coef)

    def get_absence_rate_aux_two_tables(self, m: str, t: int, s: int, criterion_field: str):
        ts = self.shift_holder.shifts_df.query(f"idx == {t}")[fld_names.EPOCH_TIMESTAMP].iloc[0]
        sh_name = self.shift_holder.get_shift_name(s)

        is_scheduled = ((self.absence_table_scheduling[fld_names.VAL_START] <= ts) &
                        (ts <= self.absence_table_scheduling[fld_names.VAL_END]) &
                        (self.absence_table_scheduling[fld_names.SHIFT_NAME] == sh_name) &
                        (self.absence_table_scheduling[fld_names.HIRING_MODALITY] == m))

        if is_scheduled.any():
            coef = self.absence_table_scheduling[is_scheduled][criterion_field].iloc[0]

        else:
            mask = ((self.absence_table[fld_names.HIRING_MODALITY] == m) &
                    (self.absence_table["shift_name"] == sh_name) &
                    (self.absence_table["day_name"] == day_name))

            coef = self.absence_table[mask][criterion_field].iloc[0]

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
    def __init__(self, df, process, stages, max_epoch_id, initial=None):
        DataHolder.__init__(self, df, process, stages, max_epoch_id, initial)


class TabulatedOutboundDataHolder(DataHolder):
    def __init__(self, df, process, stages, max_epoch_id, initial=None, subcarriers_initial=None):
        DataHolder.__init__(self, df, process, stages, max_epoch_id, initial)

        self.subcarriers_initial = subcarriers_initial

    def subcarriers_for_cpt_and_epoch(self, cpt, epoch):
        # TODO: should we inspect also the initial_backlog table looking for carriers?
        return np.unique(self.df[(self.df['cpt_idx'] == cpt) & (self.df['handling_idx'] == epoch)]['carrier'])

    def subcarrier_demand(self, subcarrier, cpt, slot):
        return int(self.df[(self.df['carrier'] == subcarrier) & (self.df['cpt_idx'] == cpt)
                           & (self.df['handling_idx'] == slot)]['count'])

    def initial_subcarrier_backlog(self, subcarrier, cpt, stage):
        return self.subcarriers_initial.get((stage, cpt), 0) if self.subcarriers_initial is not None else 0
