import logging
import os
import pandas as pd
import numpy as np

from typing import Dict, List
from pandas import DataFrame
from time import time

import kernel.data_frames_field_names as fld_names
import kernel.util.readers as readers

from kernel.data.helpers import TabulatedInboundDataHolder, Process, TabulatedOutboundDataHolder
from kernel.data.helpers import IntervalShiftHolder, ShiftHolder, DataHolder, TabulatedParameterHolder
from kernel.data.shift import TabulatedShiftMapper
from kernel.data.warehouses import WorkersParametersProcessor
from kernel.executor import RollingExecutor
from kernel.general_configurations import DevelopDumping, InputOutputPaths, Configuration, FileNames

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class RollingMain:
    def __init__(self) -> None:
        self.df_inb = pd.DataFrame()
        self.df_out = pd.DataFrame()
        self.df_inb_initial = pd.DataFrame()
        self.df_out_initial = pd.DataFrame()
        self.df_dates = pd.DataFrame()

        # Assignment "wrong ts" --> "right ts"
        self.dc_non_workable_hours = {}

    def correct_ts_to_closest_workable_hour(self, d: pd.Timestamp, handlings) -> pd.Timestamp:
        ts = self.dc_non_workable_hours.get(d, d)
        while ts not in handlings:
            ts = ts - pd.DateOffset(hours=1)
        ts += pd.DateOffset(hours=1)
        self.dc_non_workable_hours[d] = ts
        return ts

    def do_index_handlings(self) -> None:

        self.df_inb = self.df_inb.merge(self.df_dates[['date', 'idx']], left_on='handling_ts', right_on='date',
                                        how='inner')
        self.df_inb.rename(columns={'idx': 'handling_idx'}, inplace=True)
        self.df_inb.drop(columns='date', inplace=True)

        self.df_out = self.df_out.merge(self.df_dates[['date', 'idx']], left_on='handling_ts', right_on='date',
                                        how='inner')
        self.df_out.rename(columns={'idx': 'handling_idx'}, inplace=True)
        self.df_out.drop(columns='date', inplace=True)

    def do_index_cpts(self):
        self.df_inb = self.df_inb.merge(self.df_dates[['date', 'idx']], left_on='sla_ts', right_on='date', how='inner')
        self.df_inb.rename(columns={'idx': 'sla_idx'}, inplace=True)
        self.df_inb.drop(columns='date', inplace=True)

        self.df_out = self.df_out.merge(self.df_dates[['date', 'idx']], left_on='cpt_ts', right_on='date', how='inner')
        self.df_out.rename(columns={'idx': 'cpt_idx'}, inplace=True)
        self.df_out.drop(columns='date', inplace=True)

        self.df_inb_initial = self.df_inb_initial.merge(self.df_dates[['date', 'idx']], left_on='sla_ts',
                                                        right_on='date', how='inner')
        self.df_inb_initial.rename(columns={'idx': 'sla_idx'}, inplace=True)
        self.df_inb_initial.drop(columns='date', inplace=True)

        self.df_out_initial = self.df_out_initial.merge(self.df_dates[['date', 'idx']], left_on='cpt_ts',
                                                        right_on='date', how='inner')
        self.df_out_initial.rename(columns={'idx': 'cpt_idx'}, inplace=True)
        self.df_out_initial.drop(columns='date', inplace=True)

    def fix_slas_and_cpts(self, process_name):
        # This will have the repeated code inside run() to fix the cpts/slas
        pass

    def run(self) -> Dict[Process, DataFrame]:
        logger.info("Reading and pre-processing input data.\n")
        raw = readers.read_shifts()
        raw_sched = readers.read_shifts_scheduling()
        backlog_bounds = readers.read_backlog_bounds()
        polyvalents_parameters = readers.read_polyvalence_parameters()

        mapper = TabulatedShiftMapper(raw, raw_sched)

        self.df_inb, self.df_out, self.df_inb_initial, self.df_out_initial = readers.read_data()

        n_stages_inbound = 2  # TODO: these will come from wrkrs_params tables. Check short_term_formulation.py also.
        n_stages_outbound = 2

        n_modalities = readers.auxiliary_standard_read(FileNames.CONTRACT_SUBCLASSES)[
            fld_names.HIRING_MODALITY].nunique()

        # start of the model is min(fcast_in, fcast_out)
        range_min = min(self.df_inb['handling_ts'].min(), self.df_out['handling_ts'].min())
        range_max = max(self.df_inb['sla_ts'].max(), self.df_out['cpt_ts'].max())

        # dates is a bijection between indices and handlings
        self.df_dates = pd.date_range(range_min, range_max, freq='H').to_frame(name='date', index=False).reset_index(). \
            rename(columns={'index': 'idx'}).sort_values('date')

        interval = pd.Timedelta(hours=1)

        shift_holder = IntervalShiftHolder(range_min, range_max, interval, mapper, self.df_dates)

        df_presences = readers.read_presences()
        df_presences_sched_inbound, df_presences_sched_outbound = readers.read_presences_scheduling()

        df_presences_inbound = df_presences[df_presences['process'] == 'inbound']
        df_presences_outbound = df_presences[df_presences['process'] == 'outbound']

        df_absences = readers.read_absences()
        df_absences_sched = readers.read_absences_scheduling()

        df_work_forces = readers.read_workers_parameters()

        df_work_forces_sched_inbound, df_work_forces_sched_outbound = readers.read_workers_parameters_scheduling()

        # TODO: next step is to remove warehouses.py and write the clases in helpers.py
        # and also deprecate the last roster in presences and use the table directly.

        # TODO: unify the following repeated code concerning processors and null records
        inb_params_processor = WorkersParametersProcessor(shift_holder.shifts_df, raw,
                                                          df_work_forces[df_work_forces.process == 'inbound'],
                                                          df_work_forces_sched_inbound)

        out_params_processor = WorkersParametersProcessor(shift_holder.shifts_df, raw,
                                                          df_work_forces[df_work_forces.process == 'outbound'],
                                                          df_work_forces_sched_outbound)

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

        # Looking for (ts, shifts) with zero in every max_workers entry
        mask = (_null_records_inb_[fld_names.HIRING_MODALITY] == n_modalities) & (
                _null_records_inb_["stage"] == n_stages_inbound) & (
                       _null_records_inb_["max_workers"] == 0)

        _null_records_inb_ = _null_records_inb_[mask]

        mask = (_null_records_out_[fld_names.HIRING_MODALITY] == n_modalities) & (
                _null_records_out_["stage"] == n_stages_outbound) & (
                       _null_records_out_["max_workers"] == 0)

        _null_records_out_ = _null_records_out_[mask]

        _null_records_ = pd.merge(_null_records_inb_, _null_records_out_, how='inner')

        # Now iterate to erase records
        shifts_df_definitive = shift_holder.shifts_df.copy()
        for r in _null_records_.iterrows():
            # option to benchmark: use df = df[negate_condition]
            shifts_df_definitive.drop(shifts_df_definitive[(shifts_df_definitive['kind_value'] == r[1]['kind_value']) &
                                                           (shifts_df_definitive['handling_ts'] <= r[1]['end_ts']) &
                                                           (shifts_df_definitive['handling_ts'] >= r[1]['start_ts'])].
                                      index, inplace=True)

        shift_holder = ShiftHolder(shifts_df_definitive, mapper)

        shift_holder.mapper.set_dc_partial_order_of_shifts(shifts_df_definitive)

        del shifts_df_definitive

        inb_params_processor = WorkersParametersProcessor(shift_holder.shifts_df, raw,
                                                          df_work_forces[df_work_forces.process == 'inbound'],
                                                          df_work_forces_sched_inbound)

        out_params_processor = WorkersParametersProcessor(shift_holder.shifts_df, raw,
                                                          df_work_forces[df_work_forces.process == 'outbound'],
                                                          df_work_forces_sched_outbound)

        inb_params = TabulatedParameterHolder('inbound',
                                              shift_holder,
                                              backlog_bounds,
                                              inb_params_processor.process_workers_full_table,
                                              polyvalents_parameters,
                                              raw,
                                              df_presences_inbound,
                                              df_presences_sched_inbound,
                                              df_absences,
                                              df_absences_sched,
                                              Configuration.hourly_workers_cost,
                                              Configuration.hourly_work_force)

        out_params = TabulatedParameterHolder('outbound',
                                              shift_holder,
                                              backlog_bounds,
                                              out_params_processor.process_workers_full_table,
                                              polyvalents_parameters,
                                              raw,
                                              df_presences_outbound,
                                              df_presences_sched_outbound,
                                              df_absences,
                                              df_absences_sched,
                                              Configuration.hourly_workers_cost,
                                              Configuration.hourly_work_force)

        # This is the set of duty hours of the present instance
        handlings_set = set(shift_holder.shifts_df.handling_ts)

        if Configuration.fix_slas_inbound:

            if any(~self.df_inb.sla_ts.isin(shift_holder.shifts_df.handling_ts)):
                # Correct sla for non workable hours in inbound data
                df_inb_old = self.df_inb.rename(columns={'count': 'original_count',
                                                         'handling_ts': 'original_handling_ts',
                                                         'sla_ts': 'original_sla_ts'})

                self.df_inb.loc[~self.df_inb.sla_ts.isin(shift_holder.shifts_df.handling_ts), 'sla_ts'] = \
                    self.df_inb.loc[~self.df_inb.sla_ts.isin(shift_holder.shifts_df.handling_ts), 'sla_ts'].apply(
                        self.correct_ts_to_closest_workable_hour, args=(handlings_set,))

                df_inb_inconsistent = pd.concat([df_inb_old, self.df_inb['sla_ts']], axis=1)
                df_inb_inconsistent = df_inb_inconsistent.rename(columns={'sla_ts': 'corrected_sla_ts'})
                df_inb_inconsistent = df_inb_inconsistent[df_inb_inconsistent['original_sla_ts'] !=
                                                          df_inb_inconsistent['corrected_sla_ts']]
                # log warning because there are inconsistent slas in inbound data
                logger.warning('Corrected %s slas in non business hours in %s.\n',
                               str(len(df_inb_inconsistent)), FileNames.INBOUND_DATA)
                df_inb_inconsistent.to_csv(f"./warning/{FileNames.INCONSISTENT_INBOUND_CPT}")

            if any(~self.df_inb_initial.sla_ts.isin(shift_holder.shifts_df.handling_ts)):
                # Correct sla for non workable hours in inbound initial data
                df_inb_initial_old = self.df_inb_initial.rename(columns={'count': 'original_count',
                                                                         'sla_ts': 'original_sla_ts'})

                self.df_inb_initial.loc[
                    ~self.df_inb_initial.sla_ts.isin(shift_holder.shifts_df.handling_ts), 'sla_ts'] = \
                    self.df_inb_initial.loc[
                        ~self.df_inb_initial.sla_ts.isin(shift_holder.shifts_df.handling_ts), 'sla_ts'].apply(
                        self.correct_ts_to_closest_workable_hour, args=(handlings_set,))

                df_inb_initial_inconsistent = pd.concat([df_inb_initial_old, self.df_inb_initial['sla_ts']], axis=1)
                df_inb_initial_inconsistent = df_inb_initial_inconsistent.rename(columns={'sla_ts': 'corrected_sla_ts'})
                df_inb_initial_inconsistent = df_inb_initial_inconsistent[
                    df_inb_initial_inconsistent['original_sla_ts'] !=
                    df_inb_initial_inconsistent['corrected_sla_ts']]

                logger.warning('Corrected %s slas in non business hours in %s.\n',
                               str(len(df_inb_initial_inconsistent)), FileNames.INBOUND_INITIAL_DATA)

                df_inb_initial_inconsistent.to_csv(f"./warning/{FileNames.INCONSISTENT_INBOUND_INITIAL_CPT}")

        if Configuration.fix_cpts_outbound:
            # TODO: repeated code. Make an attribute of RollingMain and call for inbound and outbound.
            if any(~self.df_out.cpt_ts.isin(shift_holder.shifts_df.handling_ts)):
                # Correct cpt for non workable hours in outbound data
                df_out_old = self.df_out.rename(columns={'count': 'original_count',
                                                         'handling_ts': 'original_handling_ts',
                                                         'cpt_ts': 'original_cpt_ts'})

                self.df_out.loc[~self.df_out.cpt_ts.isin(shift_holder.shifts_df.handling_ts), 'cpt_ts'] = \
                    self.df_out.loc[~self.df_out.cpt_ts.isin(shift_holder.shifts_df.handling_ts), 'cpt_ts'].apply(
                        self.correct_ts_to_closest_workable_hour, args=(handlings_set,))

                df_out_inconsistent = pd.concat([df_out_old, self.df_out['cpt_ts']], axis=1)
                df_out_inconsistent = df_out_inconsistent.rename(columns={'cpt_ts': 'corrected_cpt_ts'})
                df_out_inconsistent = df_out_inconsistent[df_out_inconsistent['original_cpt_ts'] !=
                                                          df_out_inconsistent['corrected_cpt_ts']]
                # log warning
                logger.warning('Corrected %s cpts in non business hours in %s.\n',
                               len(df_out_inconsistent), FileNames.OUTBOUND_DATA)

                df_out_inconsistent.to_csv(f'./warning/{FileNames.INCONSISTENT_OUTBOUND_CPT}')

            if any(~self.df_out_initial.cpt_ts.isin(shift_holder.shifts_df.handling_ts)):
                # Correct cpt for non workable hours in outbound initial data
                df_out_initial_old = self.df_out_initial.rename(columns={'count': 'original_count',
                                                                         'cpt_ts': 'original_cpt_ts'})

                self.df_out_initial.loc[
                    ~self.df_out_initial.cpt_ts.isin(shift_holder.shifts_df.handling_ts), 'cpt_ts'] = \
                    self.df_out_initial.loc[
                        ~self.df_out_initial.cpt_ts.isin(shift_holder.shifts_df.handling_ts), 'cpt_ts'].apply(
                        self.correct_ts_to_closest_workable_hour, args=(handlings_set,))

                df_out_initial_inconsistent = pd.concat([df_out_initial_old, self.df_out_initial['cpt_ts']], axis=1)
                df_out_initial_inconsistent = df_out_initial_inconsistent.rename(columns={'cpt_ts': 'corrected_cpt_ts'})
                df_out_initial_inconsistent = df_out_initial_inconsistent[
                    df_out_initial_inconsistent['original_cpt_ts'] !=
                    df_out_initial_inconsistent['corrected_cpt_ts']]
                logger.warning('Corrected %s cpts in non business hours in %s.\n',
                               str(len(df_out_initial_inconsistent)), FileNames.OUTBOUND_INITIAL_DATA)
                df_out_initial_inconsistent.to_csv(f"./warning/{FileNames.INCONSISTENT_OUTBOUND_INITIAL_CPT}")

        self.do_index_handlings()
        self.do_index_cpts()

        self.df_inb.sort_values('handling_idx', inplace=True)
        self.df_out.sort_values('handling_idx', inplace=True)

        if DevelopDumping.DEV or DevelopDumping.QAS or Configuration.generate_validation_files:
            self.df_inb.to_csv(f"{InputOutputPaths.BASEDIR_VAL}/{FileNames.INBOUND_CORRECTED_SLAS}")

            self.df_inb_initial.to_csv(f"{InputOutputPaths.BASEDIR_VAL}/
                                       {FileNames.INBOUND_INITIAL_CORRECTED_SLAS}")

            self.df_out.to_csv(f"{InputOutputPaths.BASEDIR_VAL}/{FileNames.OUTBOUND_CORRECTED_SLAS}")

            self.df_out_initial.to_csv(f"{InputOutputPaths.BASEDIR_VAL}/
                                       {FileNames.OUTBOUND_INITIAL_CORRECTED_SLAS}")

            inb_params.coefficients_table.to_csv(f"{InputOutputPaths.BASEDIR_VAL}/"
                                                 f"{FileNames.INBOUND_COEFS_TABLE}", index=False)

            out_params.coefficients_table.to_csv(f"{InputOutputPaths.BASEDIR_VAL}/"
                                                 f"{FileNames.OUTBOUND_COEFS_TABLE}", index=False)

            inb_params.process_workers_full_table.to_csv(f"{InputOutputPaths.BASEDIR_VAL}/"
                                                         f"{FileNames.INBOUND_WRKRS_INFO_TABLE}", 
                                                         index=False)

            out_params.process_workers_full_table.to_csv(f"{InputOutputPaths.BASEDIR_VAL}/"
                                                         f"{FileNames.OUTBOUND_WRKRS_INFO_TABLE}", 
                                                         index=False)

            shift_holder.shifts_df.to_csv(f"{InputOutputPaths.BASEDIR_VAL}/"
                                          f"{FileNames.PARAMS_TABLE_DEFINITIVE}", index=False)

        initial = self.df_out_initial.groupby(['stage', 'cpt_idx'])['count'].sum().to_dict()
        inbound_initial = self.df_inb_initial.groupby(['stage', 'sla_idx'])['count'].sum().to_dict()
        subcarriers_initial = self.df_out_initial.groupby(
                                  ['stage', 'cpt_idx', 'carrier'])['count'].first().to_dict()

        inb_dh = TabulatedInboundDataHolder(self.df_inb, 
                                            Process.INBOUND, 
                                            2, 
                                            self.df_dates[fld_names.ID].max(), 
                                            initial=inbound_initial)

        out_dh = TabulatedOutboundDataHolder(self.df_out, 
                                             Process.OUTBOUND, 
                                             2, 
                                             self.df_dates[fld_names.ID].max(), 
                                             initial=initial,
                                             subcarriers_initial=subcarriers_initial)

        params = {Process.OUTBOUND: out_params, Process.INBOUND: inb_params}
        data_holders = {Process.OUTBOUND: out_dh, Process.INBOUND: inb_dh}

        if Configuration.make_rolling:
            shift_interval = Configuration.shift_interval
        else:
            # TODO: to define the last shift is delicate, sometimes the biggest index doesn't correspond 
            #       to the one that ends the last.
            shift_interval = shift_holder.shifts.max() + 1

        executor = RollingExecutor(shift_interval, mapper, shift_holder, data_holders, params)

        return executor.run()

if __name__ == '__main__':
    __start__ = time()
    if DevelopDumping.DEV:
        logger.debug("DEV MODE. Input from %s\n", str(InputOutputPaths.BASEDIR))
        if not os.path.isdir(InputOutputPaths.BASEDIR_OUT):
            os.makedirs(InputOutputPaths.BASEDIR_OUT)

    if DevelopDumping.QAS:
        logger.debug("Current settings:\n")
        conf = Configuration()
        logger.debug(str(conf) + "\n")

        if not os.path.isdir(InputOutputPaths.BASEDIR_OUT):
            os.makedirs(InputOutputPaths.BASEDIR_OUT)

    if DevelopDumping.DEV or Configuration.generate_validation_files:
        if not os.path.isdir(InputOutputPaths.BASEDIR_VAL):
            os.makedirs(InputOutputPaths.BASEDIR_VAL)

    try:
        # __len__, __bool__, __iter__
        logger.info('AlexTroN Version 1.2.1 March 2021\n')  # I don't like the timestamp here.
        logger.info("Read all the messages in %s and in this window. Then press ENTER to close.\n",
                    FileNames.LOG_FILE)

        RollingMain().run()
        input(f"Completed!\n\nTotal execution time was {time() - __start__:.4f} seconds."
               "Press ENTER to close. ")
    except Exception as ex:
        logger.exception('\n\n--------------------------------------------------------\n'
                         'There were errors. Read the following and the log files.\n'
                         '--------------------------------------------------------\n\n%s', ex)
        input(f"Total execution time was {time() - __start__:.4f} seconds. Press ENTER to close. ")
