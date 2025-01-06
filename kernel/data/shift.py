import calendar
import pandas as pd

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict

import kernel.data_frames_field_names as fld_names

from kernel.general_configurations import DevelopDumping, InputOutputPaths, FileNames
from kernel.util import readers

"""
Docstring on what the fuck is a 'shift' in AlexTroN:

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

# TODO figure out how to read csvs only ONCE
# TODO how to put this stuff in a class? These two are also
#      in TabulatedShiftMapper in the present module
raw = readers.read_shifts()
raw_sched = readers.read_shifts_scheduling()

if len(raw_sched):
    shift_names = pd.concat([raw['shift_name'], raw_sched['shift_name']]).unique().tolist()
else:
    shift_names = raw['shift_name'].unique().tolist()

shift_names_and_indices = dict(zip(shift_names, range(len(shift_names))))

ShiftKind = Enum('ShiftKind', shift_names_and_indices)


@dataclass(frozen=True, eq=True)
class Shift:
    day: int
    kind: ShiftKind

    def day_name(self) -> str:
        return calendar.day_name[self.day]


class TabulatedShiftMapper:
    def __init__(self, data, data_scheduled) -> None:
        self.data = data  # This is the raw_shifts
        self.data_scheduled = data_scheduled  # This is the raw_shifts_scheduled
        self.reschedule_records = pd.DataFrame()
        self.active_records = pd.DataFrame()

        self.dc_partial_order_of_shifts: Dict[str, List[str]] = {}

    def map_shifts(self, date: pd.Timestamp) -> List[Tuple[Shift, str, int, str]]:
        """ Mainly used in helpers::ShiftHolder.__get_idxs
            to build the shifts_df table
            This requires a time slot.
            Also used in put_friendly_fields() and in correct_slas() """

        dow = date.dayofweek
        answer = []

        day_shifts = self.data[self.data['day_name'] == calendar.day_abbr[dow]]
        for i in range(len(day_shifts)):

            # FLAG = 0

            start, end, shift_name, work_day_name = day_shifts.iloc[i][['start', 'end', 'shift_name', 'day_name']]

            if start <= date.hour < end:
                answer.append((Shift(dow, ShiftKind[shift_name]), 0, work_day_name))

            if end < start <= date.hour:
                answer.append((Shift(dow, ShiftKind[shift_name]), 0, work_day_name))

        # is this ts in any across--midnight shift the day before?
        day_before_shifts = self.data[self.data['day_name'] == calendar.day_abbr[(dow - 1) % 7]]

        for j in range(len(day_before_shifts)):

            start, end, shift_name, work_day_name = day_before_shifts.iloc[j][['start',
                                                                               'end',
                                                                               'shift_name',
                                                                               'day_name']]

            if date.hour < end < start:
                answer.append((Shift((dow - 1) % 7, ShiftKind[shift_name]), 1, work_day_name))

        return answer

    def schedule_flag(self, st):
        # This is to classify a shift as 'scheduled' only if it is a NEW shift.
        if st not in self.data['shift_name'].unique():
            ret = 1
        elif st in self.reschedule_records['shift_name'].unique():
            # The intersection
            ret = 2
        else:
            ret = 0
        return ret

    def map_scheduled_shifts(self, df, index_handling_bijection):
        # This function is about the shifts listed specifically in shifts_parameters_scheduling.csv
        new_df_recs = []
        index_handling_bijection = index_handling_bijection.groupby('date')['idx'].first().to_dict()

        if len(self.data_scheduled):
            # The regular shifts rescheduled. The following line is like sets intersection.
            self.reschedule_records = self.data_scheduled[self.data_scheduled.shift_name.isin(self.data['shift_name'])]
            # Traverse each record with an active shift. Shifts with -1 -1 don't add any record.
            self.active_records = self.data_scheduled[self.data_scheduled.start != self.data_scheduled.end]

        # First wipe the old records out, then treat the rescheduled as if they were completely new.
        for row in self.reschedule_records.itertuples():
            for _day_ in pd.date_range(start=row.validity_start, end=row.validity_end):
                df = df.drop(df[(df.shift_name == row.shift_name) & (df.calendar_reference_day == _day_)].index)

        for row in self.active_records.itertuples():
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
                    if index:
                        # Now we write a record for each hour in the present new shift
                        new_df_recs.append({'work_day_name': _day_.day_name()[0:3],
                                            'handling_ts': ts,
                                            'idx': index_handling_bijection[ts],
                                            'shifts_for_ts': Shift(_day_.dayofweek, ShiftKind[row.shift_name]),
                                            'calendar_reference_day': pd.Timestamp(_day_.year, _day_.month, _day_.day),
                                            'kind_value': ShiftKind[row.shift_name].value,
                                            'shift_name': ShiftKind[row.shift_name].name,
                                            'is_scheduled': self.schedule_flag(row.shift_name)})

        if DevelopDumping.DEV:
            self.reschedule_records.to_csv(f"{InputOutputPaths.BASEDIR_OUT}/{FileNames.RESCHEDULE_RECORDS}")
            df.to_csv(f"{InputOutputPaths.BASEDIR_OUT}/{FileNames.DF_WIPED_OUT}")
            self.active_records.to_csv(f"{InputOutputPaths.BASEDIR_OUT}/{FileNames.ACTIVE_RECORDS}")
            pd.DataFrame(new_df_recs).to_csv(f"{InputOutputPaths.BASEDIR_OUT}/{FileNames.NEW_DF_RECS}")

        return pd.concat([df, pd.DataFrame(new_df_recs)], ignore_index=True)

    def set_dc_partial_order_of_shifts(self, df_from: pd.DataFrame) -> None:
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

        self.dc_partial_order_of_shifts = po

        if DevelopDumping.DEV:
            with open(f"{InputOutputPaths.BASEDIR_VAL}/{FileNames.PARTIAL_ORDER_OF_SHIFTS}", 'w') as file_po:
                file_po.write(str(self.dc_partial_order_of_shifts))
