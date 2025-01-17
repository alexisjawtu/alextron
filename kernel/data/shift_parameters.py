import calendar
import pandas as pd
import numpy as np
import logging

from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List, Dict, Any

import kernel.util.readers as readers
import kernel.data_frames_field_names as fld_names

from kernel.general_configurations import DevelopDumping, InputOutputPaths, FileNames
from kernel.util.readers import imported_shift_names_and_indices
from kernel.data.warehouses import WorkersParametersProcessor

day_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def explode_modality_all(mod_name_by_shift_name, df):
    # Function that replaces the modality "ALL" with the modalities associated with that shift
    # Requires field names HIRING_MODALITY and SHIFT_NAME in the both tables
    if not df.empty:
        df_all = df.loc[
            ((df[fld_names.HIRING_MODALITY]).str.upper() == fld_names.ALL_MODALITY)]
        df.drop(df_all.index, inplace=True)

        if not df_all.empty:
            df_all = df_all.drop(columns=fld_names.HIRING_MODALITY)
            df_all = df_all.merge(mod_name_by_shift_name, how='inner', on=fld_names.SHIFT_NAME)
            df_all = df_all.explode(fld_names.HIRING_MODALITY)
            df = df.append(df_all)
    return df


@dataclass
class ProcessSignature:
    """ Elementary 'static' data concerning a process in a site.

    This class is aimed to deprecate de Process(Enum)
    when we get to the point to have arbitrary processes and stages.

    Complete with more fields as necessary.

    Think if it is better to read all directly from the data_holders.
    """
    index: int  # Previously corresponding to the enum.
    name: str
    stage_names: Dict


"""
Docstring on what the fuck is a 'shift' in moletron:

We have different things in the program concerning a shift, namely 
a 'shift_name', a 'shift_type', a 'shift_number', a class 'ShiftKind' and a class 'Shift':

    - shift_name: str (comes with the input) 'AFTERNOON3W2'
    - shift_type: str (comes with the input) 'AFTERNOON3'

            Note: this string shift_type was born with the feature of hiring and transfers.

    - shift_number: int (unique indexation with integers to run the exercise) 

            [2021-nov-19 'AFTERNOON3W2'] <--> 14
            [2021-nov-19 'NIGHT0W2']     <--> 15

    - ShiftKind (a struct based on an Enum made with a shift_name and a unique int --not the shift_number!)

            This is simply a bijection between human words designing a shift_name 
            and an integer, inner to the program, not for any user.

            Example:

                    given a list of shift_name ['MORNING0W2', 'AFTERNOON1W1', 'NIGHT1W3'], 
                    then ShiftKind is the following Enum (enumeration):

                                  <ShiftKind.MORNING0W2: 0>
                                  <ShiftKind.AFTERNOON1W1: 1>
                                  <ShiftKind.NIGHT1W3: 2>

                    If we do, for example:

                        >>> a = ShiftKind(0)  

                        or, equivalently

                        >>> a = ShiftKind.MORNING0W2

                        or, equivalently

                        >>> a = ShiftKind["MORNING0W2"]

                    then:

                        a.name == 'MORNING0W2'
                        a.value == 0

                    so we could say that " a is the MORNING0W2 ".

    - Shift (a struct based on a dataclass with only two attibutes)

            day: int (day of the week, from 0 to 6, starting with monday <--> 0)
            kind: ShiftKind (which ShiftKind we are looking at during that day)

            Example:

                    if we want to instantiate the NIGHT1 in the wednesday of the second week,
                    then we construct the following object

                    >>> sh = Shift(2, ShiftKind.NIGHT1W2)

Important: the transfer, hiring, dismissal and unitary costs will be in helpers.ShiftHolder
           --not in ParameterHolder, because they are global constants.
"""

"""
Initial Version:

# TODO figure out how to read csvs only ONCE
# TODO how to put this stuff in a class? These two are also
# in TabulatedShiftMapper in the present module
raw = readers.read_shifts()
raw_sched = readers.read_shifts_scheduling()

if len(raw_sched):
    shift_names = pd.concat([raw['shift_name'], raw_sched['shift_name']]).unique().tolist()
else:
    shift_names = raw['shift_name'].unique().tolist()

shift_names_and_indices = dict(zip(shift_names, range(len(shift_names))))

ShiftKind = Enum('ShiftKind', shift_names_and_indices)
"""


class ShiftParametersGenerator:
    def __init__(self,
                 read_data: readers.MoletronInput,
                 range_min: pd.Timestamp,
                 range_max: pd.Timestamp) -> None:

        self.data = read_data.raw_shifts  # This is the raw_shifts
        self.data_scheduled = read_data.raw_shifts_scheduled  # This is the raw_shifts_scheduled

        # shift_holder
        # df_dates
        # dates is a bijection between indices and handlings
        self.df_dates = pd.date_range(range_min, range_max, freq='H').to_frame(name='date',
                                                                               index=False).reset_index(). \
            rename(columns={'index': 'idx'})

        interval = pd.Timedelta(hours=1)

        self.map_shifts = self.set_map_shifts(range_min, range_max, interval)

        records = []
        ts = range_min
        idx = 0

        while ts <= range_max:
            s = self.map_shifts[ts]
            # This is the df for ShiftHolder.shifts_df
            records += [{'work_day_name': work_day_name,
                         'day_of_week': dow,
                         'handling_ts': ts,
                         'idx': idx,
                         'calendar_reference_day': pd.Timestamp(ts.year, ts.month, ts.day) - pd.Timedelta(days=b),
                         'kind_value': shift_kind_value,
                         'shift_name': shift_kind_name,
                         'is_scheduled': 0,
                         fld_names.START_MINS: ratio_start,
                         fld_names.END_MINS: ratio_end}
                        for dow, shift_kind_name, shift_kind_value, b, work_day_name, ratio_start, ratio_end in s]

            idx += 1
            ts += interval

        df = pd.DataFrame.from_records(records)
        df = self.map_scheduled_shifts(df, self.df_dates)
        self.shifts_df = df.sort_values('handling_ts')

        # The following line calculates and sets the unique integer shift_idx.
        self.shifts_df['shift_idx'] = self.shifts_df.groupby(['calendar_reference_day', 'kind_value']).ngroup()

        # to generate all the combinations (shift by modality) necessary to rewrite the modalities declared with ALL
        mod_name_by_shift_name = read_data.df_shift_contract_modality.groupby(fld_names.SHIFT_NAME).agg(list)

        df_presences = explode_modality_all(mod_name_by_shift_name, read_data.df_presences)
        self.df_presences_sched_inbound = explode_modality_all(mod_name_by_shift_name,
                                                               read_data.df_presences_sched_inbound)
        self.df_presences_sched_outbound = explode_modality_all(mod_name_by_shift_name,
                                                                read_data.df_presences_sched_outbound)

        self.df_presences_inbound = df_presences[df_presences['process'] == 'inbound']
        self.df_presences_outbound = df_presences[df_presences['process'] == 'outbound']

        self.df_absences = explode_modality_all(mod_name_by_shift_name, read_data.df_absences)
        self.df_absences_sched = explode_modality_all(mod_name_by_shift_name, read_data.df_absences_sched)
        self.df_work_forces = explode_modality_all(mod_name_by_shift_name, read_data.df_work_forces)
        self.df_work_forces_sched_inbound = explode_modality_all(mod_name_by_shift_name,
                                                                 read_data.df_work_forces_sched_inbound)
        self.df_work_forces_sched_outbound = explode_modality_all(mod_name_by_shift_name,
                                                                  read_data.df_work_forces_sched_outbound)
        self.polyvalents_parameters = explode_modality_all(mod_name_by_shift_name, read_data.polyvalents_parameters)
        # TODO: next step is to remove warehouses.py and write the clases in helpers.py
        #  and also deprecate the last roster in presences and use the table directly.
        #
        # TODO: unify the following repeated code concerning processors and null records
        inb_params_processor = WorkersParametersProcessor(self.shifts_df, read_data.raw_shifts,
                                                          self.df_work_forces[self.df_work_forces.process == 'inbound'],
                                                          self.df_work_forces_sched_inbound)

        out_params_processor = WorkersParametersProcessor(self.shifts_df, read_data.raw_shifts,
                                                          self.df_work_forces[
                                                              self.df_work_forces.process == 'outbound'],
                                                          self.df_work_forces_sched_outbound)

        _null_records_inb_ = inb_params_processor.process_workers_full_table.groupby(
            ['kind_value', 'start_ts', 'end_ts'])[['max_workers', fld_names.HIRING_MODALITY, 'stage']].agg(
            {fld_names.HIRING_MODALITY: lambda t: len(np.unique(t)),
             'stage': lambda t: len(np.unique(t)),
             'max_workers': sum}).reset_index()

        _null_records_out_ = out_params_processor.process_workers_full_table.groupby(
            ['kind_value', 'start_ts', 'end_ts'])[['max_workers', fld_names.HIRING_MODALITY, 'stage']].agg(
            {fld_names.HIRING_MODALITY: lambda t: len(np.unique(t)),
             'stage': lambda t: len(np.unique(t)),
             'max_workers': sum}).reset_index()

        # TODO: the following will come from wrkrs_params tables. Think a good way
        # to catch it all, for example in a Dict, in a version with arbitrary processes/stages.
        self.inbound_signature = ProcessSignature(0, "INBOUND", {0: "receiving", 1: "checkin"})
        self.outbound_signature = ProcessSignature(0, "OUTBOUND", {0: "picking", 1: "packing"})
        self.n_stages_inbound: int = len(self.inbound_signature.stage_names)
        self.n_stages_outbound: int = len(self.outbound_signature.stage_names)
        self.n_modalities = read_data.modalities[fld_names.HIRING_MODALITY].nunique()

        # Looking for (ts, shifts) with zero in every max_workers entry
        mask = (_null_records_inb_[fld_names.HIRING_MODALITY] == self.n_modalities) & (
                _null_records_inb_["stage"] == self.n_stages_inbound) & (
                       _null_records_inb_["max_workers"] == 0)

        _null_records_inb_ = _null_records_inb_[mask]

        mask = (_null_records_out_[fld_names.HIRING_MODALITY] == self.n_modalities) & (
                _null_records_out_["stage"] == self.n_stages_outbound) & (
                       _null_records_out_["max_workers"] == 0)

        _null_records_out_ = _null_records_out_[mask]

        _null_records_ = pd.merge(_null_records_inb_, _null_records_out_, how='inner')

        # Now iterate to erase records
        shifts_df_definitive = self.shifts_df.copy()
        for r in _null_records_.iterrows():
            # option to benchmark: use df = df[negate_condition]
            shifts_df_definitive.drop(shifts_df_definitive[(shifts_df_definitive['kind_value'] == r[1]['kind_value']) &
                                                           (shifts_df_definitive['handling_ts'] <= r[1]['end_ts']) &
                                                           (shifts_df_definitive['handling_ts'] >= r[1]['start_ts'])].
                                      index, inplace=True)

        self.shifts_df = shifts_df_definitive.sort_values('handling_ts')

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

        self.dict_days_quantity_per_shift = shifts_df_definitive.groupby([fld_names.SHIFT_NAME])[
            fld_names.WORK_DAY_NAME].nunique().to_dict()

        # moletron

        self.modalities_by_shift_idx = pd.merge(self.shifts_df[['shift_idx', 'shift_name']].drop_duplicates(),
                                                read_data.df_shift_contract_modality[
                                                    ['contract_modality', 'shift_name']]
                                                .drop_duplicates())
        self.dc_modalities_by_shift_idx = self.modalities_by_shift_idx.groupby('shift_idx')['contract_modality'].apply(
            list).to_dict()
        self.dc_shift_idx_by_modalities = self.modalities_by_shift_idx.groupby('contract_modality')['shift_idx'].apply(
            list).to_dict()

        self.dc_partial_order_of_shifts = self.set_dc_partial_order_of_shifts(shifts_df_definitive)

        del shifts_df_definitive

        self.inb_params_processor = WorkersParametersProcessor(self.shifts_df, read_data.raw_shifts,
                                                               self.df_work_forces[
                                                                   self.df_work_forces.process == 'inbound'],
                                                               self.df_work_forces_sched_inbound)

        self.out_params_processor = WorkersParametersProcessor(self.shifts_df, read_data.raw_shifts,
                                                               self.df_work_forces[
                                                                   self.df_work_forces.process == 'outbound'],
                                                               self.df_work_forces_sched_outbound)

    def set_map_shifts(self, range_min: pd.Timestamp, range_max: pd.Timestamp, interval: pd.Timedelta) -> Dict:
        """ Mainly used in helpers::ShiftHolder.__get_idxs
            to build the shifts_df table
            This requires a time slot.
            Also used in put_friendly_fields() and in correct_slas() """

        map_shift = {}

        ts = range_min
        while ts <= range_max:
            dow = ts.dayofweek
            answer = []

            day_shifts = self.data[self.data[fld_names.DAY] == calendar.day_abbr[dow]]
            for i in range(len(day_shifts)):

                # FLAG = 0
                start, end, shift_name, work_day_name, ratio_start, ratio_end = day_shifts.iloc[i][[
                    "start",
                    "end",
                    "shift_name",
                    fld_names.DAY,
                    fld_names.START_MINS,
                    fld_names.END_MINS
                ]]

                if start <= ts.hour < end:
                    answer.append((dow, shift_name, imported_shift_names_and_indices[shift_name], 0, work_day_name,
                                   ratio_start, ratio_end))

                if end < start <= ts.hour:
                    answer.append((dow, shift_name, imported_shift_names_and_indices[shift_name], 0, work_day_name,
                                   ratio_start, ratio_end))

            # is this ts in any across--midnight shift the day before?
            day_before_shifts = self.data[self.data[fld_names.DAY] == calendar.day_abbr[(dow - 1) % 7]]

            for j in range(len(day_before_shifts)):

                start, end, shift_name, work_day_name, ratio_start, ratio_end = day_before_shifts.iloc[j][[
                    "start",
                    "end",
                    "shift_name",
                    fld_names.DAY,
                    fld_names.START_MINS,
                    fld_names.END_MINS
                ]]

                if ts.hour < end < start:
                    answer.append((
                        (dow - 1) % 7, shift_name, imported_shift_names_and_indices[shift_name], 1, work_day_name,
                        ratio_start, ratio_end))

            map_shift[ts] = answer
            ts += interval

        return map_shift

    def map_scheduled_shifts(self, df, index_handling_bijection):
        # This function is about the shifts listed specifically in shifts_parameters_scheduling.csv
        new_df_recs = []
        index_handling_bijection = index_handling_bijection.groupby('date')['idx'].first().to_dict()
        active_records = pd.DataFrame()
        reschedule_records = pd.DataFrame()

        if len(self.data_scheduled):
            # The regular shifts rescheduled. The following line is like sets intersection.
            reschedule_records = self.data_scheduled[
                self.data_scheduled.shift_name.isin(self.data[fld_names.SHIFT_NAME])]
            # Traverse each record with an active shift. Shifts with -1 -1 don't add any record.

            active_records = self.data_scheduled[self.data_scheduled.start != self.data_scheduled.end]

        # First wipe the old records out, then treat the rescheduled as if they were completely new.
        for row in reschedule_records.itertuples():
            for _day_ in pd.date_range(start=row.validity_start, end=row.validity_end):
                df = df.drop(df[(df.shift_name == row.shift_name) & (df.calendar_reference_day == _day_)].index)

        for row in active_records.itertuples():
            # The following idea treats shifts within a day and across midnight also
            shift_size = (row.end - row.start) % 24
            # For each day within the validity, there is a new shift
            for _day_ in pd.date_range(start=row.validity_start, end=row.validity_end):

                start_ts = _day_ + pd.Timedelta(hours=row.start)
                end_ts = _day_ + pd.Timedelta(hours=row.start + shift_size - 1)

                # TODO: Add more hours to index bijection list
                for ts in pd.date_range(start=start_ts, end=end_ts, freq='H'):
                    # Check if shift timestamp is within the planning horizon
                    index = index_handling_bijection.get(ts)
                    if not index == None:
                        # Now we write a record for each hour in the present new shift
                        new_df_recs.append({'work_day_name': _day_.day_name()[0:3],
                                            'day_of_week': _day_.dayofweek,
                                            'handling_ts': ts,
                                            'idx': index_handling_bijection[ts],
                                            'calendar_reference_day': pd.Timestamp(_day_.year, _day_.month, _day_.day),
                                            'kind_value': imported_shift_names_and_indices[row.shift_name],
                                            'shift_name': row.shift_name,
                                            'is_scheduled': self.schedule_flag(reschedule_records, row.shift_name),
                                            fld_names.START_MINS: row.start_minute,
                                            fld_names.END_MINS: row.end_minute})

        if DevelopDumping.DEV:
            reschedule_records.to_csv(f"{InputOutputPaths.BASEDIR_OUT}/{FileNames.RESCHEDULE_RECORDS}")
            df.to_csv(f"{InputOutputPaths.BASEDIR_OUT}/{FileNames.DF_WIPED_OUT}")
            active_records.to_csv(f"{InputOutputPaths.BASEDIR_OUT}/{FileNames.ACTIVE_RECORDS}")
            pd.DataFrame(new_df_recs).to_csv(f"{InputOutputPaths.BASEDIR_OUT}/{FileNames.NEW_DF_RECS}")

        return pd.concat([df, pd.DataFrame(new_df_recs)], ignore_index=True)

    def schedule_flag(self, reschedule_records, st):
        # This is to classify a shift as 'scheduled' only if it is a NEW shift.
        if st not in self.data[fld_names.SHIFT_NAME].unique():
            ret = 1
        elif st in reschedule_records[fld_names.SHIFT_NAME].unique():
            # The intersection
            ret = 2
        else:
            ret = 0
        return ret

    def set_dc_partial_order_of_shifts(self, df_from: pd.DataFrame) -> dict:
        """
        If we have the following hierarchy in the input:

               shift_type    shift_name  min_epoch_global_idx
            0  AFTERNOON0  AFTERNOON0W3                    60
            1  AFTERNOON0  AFTERNOON0W2                    53
            2  AFTERNOON0  AFTERNOON0W1                    29
            3    MORNING0    MORNING0W2                    26
            4    MORNING0    MORNING0W1                    21
            5    MORNING1    MORNING1W2                    26
            6    MORNING1    MORNING1W1                    21

        this method gives the following partial order dict:

            {
                  'AFTERNOON0': array(['AFTERNOON0W1', 'AFTERNOON0W2', 'AFTERNOON0W3'], dtype=object),
                  'AFTERNOON1': array(['AFTERNOON1W1', 'AFTERNOON1W2', 'AFTERNOON1W3'], dtype=object),
                  'MORNING0': array(['MORNING0W1', 'MORNING0W2'], dtype=object),
                  'MORNING1': array(['MORNING1W1', 'MORNING1W2'], dtype=object),
            }

        so we traverse the lists and get the shift_names in order.

        An option to study, perhaps, because the number of cases is small:

            _keys_ = [a for a, _ in series_res.keys()]
            _keys_ = list(OrderedDict.fromkeys(_keys_))
            for a in _keys_:
                 for k in series_res.get(a).keys():
                    print(k)

        """
        po = {}

        a = df_from[['shift_name', 'idx']].merge(self.data[['shift_name', 'shift_type']], on='shift_name').groupby(
            ['shift_type', 'shift_name']).agg({'idx': min}).reset_index()
        gb = a.groupby('shift_type')

        for g in gb.groups:
            curr = gb.get_group(g).sort_values('idx', ascending=True)
            po[g] = curr.shift_name.to_numpy()

        if DevelopDumping.DEV:
            with open(f"{InputOutputPaths.BASEDIR_VAL}/{FileNames.PARTIAL_ORDER_OF_SHIFTS}", 'w') as file_po:
                file_po.write(str(po))

        return po

    def get_days_quantity_per_shift(self, shift_name: str):
        return self.dict_days_quantity_per_shift[shift_name]

    def get_shift_name(self, s: int) -> str:
        return self.shifts_df.query("shift_idx == @s")["shift_name"].iloc[0]

    def shifts_for_epoch_range(self, start: int, end: int) -> List:
        data = self.shifts_df.query('@start <= idx <= @end and shift_idx >= 0')['shift_idx'].sort_values().unique()
        return [int(s) for s in data]

    def get_cost(self, shift_name: str, modality: str, cost_name: str) -> float:
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
        bool_mask = (self.df_workers_costs[fld_names.SHIFT_NAME] == shift_name) & \
                    (self.df_workers_costs[fld_names.HIRING_MODALITY] == modality)

        if bool_mask.any():
            one_row_df = self.df_workers_costs[bool_mask]

            series = one_row_df[cost_name]  # requesting a column gives a Series, not a DataFrame.

            cost = series.loc[one_row_df.index.min()]  # the first index is the only index.

        else:
            logger.warning("Missing cost for shift %s with modality %s,\n"
                           "we will use the value 1 and the instance will keep running.\n"
                           "Maybe you forgot to put it in file %s?\n", shift_name, modality,
                           FileNames.WORKERS_COSTS)
            cost = 1

        return float(cost)
