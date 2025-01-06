import pandas as pd

import sot_fbm_staffing.data_frames_field_names as fld_names


class WorkersParametersProcessor: 
    def __init__(
        self, 
        shifts_df: pd.DataFrame, 
        raw_shifts: pd.DataFrame, 
        df_work_forces: pd.DataFrame,
        df_work_forces_sched: pd.DataFrame
    ) -> None:

        self.shifts_df = shifts_df
        self.raw_shifts = raw_shifts
        self.df_work_forces = df_work_forces
        self.df_work_forces_sched = df_work_forces_sched
        self.process_workers_full_table = pd.DataFrame()

        table_full = self.shifts_df.copy()
        table_full[fld_names.AUX_TS] = table_full[fld_names.EPOCH_TIMESTAMP]
        table_full = table_full[
            [fld_names.SHIFT_ID, 
             fld_names.SHIFT_NAME, 
             'kind_value', 
             'is_scheduled', 
             fld_names.EPOCH_TIMESTAMP, 
             fld_names.AUX_TS]
        ]

        # we add two columns defining the time interval, so that we can assign to the right validity day.
        # Warning: the 'max' aggregation in the following gives t <= end_ts. Remember that in the input we have
        # t < end for shift limits.
        table_full = table_full.groupby([
            fld_names.SHIFT_ID, 
            fld_names.SHIFT_NAME, 
            'kind_value', 
            'is_scheduled'
        ])[[fld_names.EPOCH_TIMESTAMP, fld_names.AUX_TS]].agg({fld_names.EPOCH_TIMESTAMP: min, fld_names.AUX_TS: max})

        table_full = table_full.reset_index().rename(columns={
            fld_names.EPOCH_TIMESTAMP: 'start_ts', 
            fld_names.AUX_TS: 'end_ts'
        })

        # Split into cases, regular or scheduled
        # Is_scheduled == 1 means it was speciefied within shifts_parameters_scheduling.csv
        # We may also have 'scheduled changes' but in regular shifts
        table = table_full[(table_full.is_scheduled == 0) | (table_full.is_scheduled == 2)]
        table_sched = table_full[table_full.is_scheduled == 1]

        # non-scheduled only
        workers_table = table.merge(df_work_forces).drop(columns=fld_names.PROCESS)

        # This loop is for 'scheduled changes' in regular shifts
        if len(df_work_forces_sched):
            # The following loop repeats cases. We may put three different flags
            for row in df_work_forces_sched[df_work_forces_sched.shift_name.
                                            isin(raw_shifts[fld_names.SHIFT_NAME])].itertuples():
                # Add one more day to validity_end because we don't consider hours anymore
                ts_validity_end: pd.Timestamp = row.validity_end + pd.Timedelta(days=1)

                # First option, a quadratic cycle. Then we may only explore the records in the inner loop with
                # certain shift_names only, and third option is explore specific merge and combines in pandas
                for row_target in workers_table.itertuples():
                    # TODO: make this 'if' more pandanic, more DRY and more readable
                    if row.validity_start <= row_target.start_ts < ts_validity_end and \
                       row.shift_name == row_target.shift_name and \
                       row.stage == row_target.stage and \
                       row.contract_modality == row_target.contract_modality:

                        workers_table.loc[row_target.Index, fld_names.WRK_FORCE] = row.work_force
                        workers_table.loc[row_target.Index, fld_names.MAX_WRKRS] = row.max_workers

        # scheduled in shifts_parameters_scheduling.csv only
        workers_table_sched = pd.DataFrame()
        if len(table_sched) and len(df_work_forces_sched):
            workers_table_sched = table_sched.merge(df_work_forces_sched)
            # keep only those intervals containing the shifts
            workers_table_sched.query('validity_start <= start_ts and end_ts <= validity_end', inplace=True)
            workers_table_sched.drop(columns=['process', 'validity_start', 'validity_end'], inplace=True)

        # The following table is for the <<experts>> (formerly misnamed "permanents") of any contract modality.
        self.process_workers_full_table = pd.concat([workers_table, workers_table_sched], ignore_index=True)
