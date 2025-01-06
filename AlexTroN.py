import logging
import os
import pandas as pd
import numpy as np

from typing import Dict, List
from pandas import DataFrame
from time import time

import kernel.data_frames_field_names as fld_names
import kernel.util.readers as readers

from kernel.data.helpers import (
    Process,
    TabulatedParameterHolder,
    TabulatedInboundDataHolder,
    TabulatedOutboundDataHolder
)
from kernel.data.shift_parameters import ShiftParametersGenerator
from kernel.executor import RollingExecutor
from kernel.general_configurations import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def explode_modality_all(mod_name_by_shift_name, df):
    TODO:    WRITE THIS MYSELF
    # Function that replaces the modality "ALL" with the modalities associated with that shift
    # Requires field names HIRING_MODALITY and SHIFT_NAME in both tables
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


class RollingMain:
    def __init__(self) -> None:
        self.df_inb: pd.DataFrame = None
        self.df_out: pd.DataFrame = None
        self.df_inb_initial: pd.DataFrame = None
        self.df_out_initial: pd.DataFrame = None
        # self.df_dates: pd.DataFrame = None
        self.raw_eh_parameters: pd.DataFrame = None
        # Assignment "wrong ts" --> "right ts"
        self.dc_non_workable_hours: Dict = {}

    def correct_ts_to_closest_workable_hour(self, d: pd.Timestamp, handlings) -> pd.Timestamp:
        ts = self.dc_non_workable_hours.get(d, d)
        while ts not in handlings:
            ts = ts - pd.DateOffset(hours=1)
        ts += pd.DateOffset(hours=1)
        self.dc_non_workable_hours[d] = ts
        return ts

    def do_index_handlings(self, df_dates) -> None:

        self.df_inb = self.df_inb.merge(df_dates[['date', 'idx']], left_on='handling_ts', right_on='date',
                                        how='inner')
        self.df_inb.rename(columns={'idx': 'handling_idx'}, inplace=True)
        self.df_inb.drop(columns='date', inplace=True)

        self.df_out = self.df_out.merge(df_dates[['date', 'idx']], left_on='handling_ts', right_on='date',
                                        how='inner')
        self.df_out.rename(columns={'idx': 'handling_idx'}, inplace=True)
        self.df_out.drop(columns='date', inplace=True)

    def do_index_cpts(self, df_dates):
        self.df_inb = self.df_inb.merge(df_dates[['date', 'idx']], left_on='sla_ts', right_on='date', how='inner')
        self.df_inb.rename(columns={'idx': 'sla_idx'}, inplace=True)
        self.df_inb.drop(columns='date', inplace=True)

        self.df_out = self.df_out.merge(df_dates[['date', 'idx']], left_on='cpt_ts', right_on='date', how='inner')
        self.df_out.rename(columns={'idx': 'cpt_idx'}, inplace=True)
        self.df_out.drop(columns='date', inplace=True)

        self.df_inb_initial = self.df_inb_initial.merge(df_dates[['date', 'idx']], left_on='sla_ts',
                                                        right_on='date', how='inner')
        self.df_inb_initial.rename(columns={'idx': 'sla_idx'}, inplace=True)
        self.df_inb_initial.drop(columns='date', inplace=True)

        self.df_out_initial = self.df_out_initial.merge(df_dates[['date', 'idx']], left_on='cpt_ts',
                                                        right_on='date', how='inner')
        self.df_out_initial.rename(columns={'idx': 'cpt_idx'}, inplace=True)
        self.df_out_initial.drop(columns='date', inplace=True)

    def fix_slas_and_cpts(self, process_name):
        # This will have the repeated code inside run() to fix the cpts/slas
        pass

    def run(self) -> Dict[Process, DataFrame]:
        logger.info("Reading and pre-processing input data.\n")
        delta_extra_hours = 0
        if Configuration.activate_extra_hours:
            logger.info("Extra hours are enabled.\n")
            self.raw_eh_parameters = readers.read_extra_hours_parameters()
            delta_extra_hours = self.raw_eh_parameters["max_daily_extra_hours"].max()
        else:
            logger.info("Extra hours are disabled.\n")

        data = readers.Input()

        self.df_inb, self.df_out, self.df_inb_initial, self.df_out_initial = readers.read_data()
        range_min = min(self.df_inb['handling_ts'].min(),
                        self.df_out['handling_ts'].min()) - pd.Timedelta(hours=delta_extra_hours)
        range_max = max(self.df_inb['sla_ts'].max(),
                        self.df_out['cpt_ts'].max()) + pd.Timedelta(hours=delta_extra_hours)

        shift_params = ShiftParametersGenerator(data, range_min, range_max)

        inb_params = TabulatedParameterHolder('inbound',
                                              shift_params,
                                              data.backlog_bounds,
                                              shift_params.inb_params_processor.process_workers_full_table,
                                              data.polyvalents_parameters,
                                              data.raw_shifts,
                                              shift_params.df_presences_inbound,
                                              shift_params.df_presences_sched_inbound,
                                              shift_params.df_absences,
                                              shift_params.df_absences_sched,
                                              Configuration.hourly_workers_cost,
                                              Configuration.hourly_work_force)

        out_params = TabulatedParameterHolder('outbound',
                                              shift_params,
                                              data.backlog_bounds,
                                              shift_params.out_params_processor.process_workers_full_table,
                                              data.polyvalents_parameters,
                                              data.raw_shifts,
                                              shift_params.df_presences_outbound,
                                              shift_params.df_presences_sched_outbound,
                                              shift_params.df_absences,
                                              shift_params.df_absences_sched,
                                              Configuration.hourly_workers_cost,
                                              Configuration.hourly_work_force)

        # This is the set of duty hours of the present instance
        handlings_set = set(shift_params.shifts_df.handling_ts)

        if Configuration.fix_slas_inbound:

            if any(~self.df_inb.sla_ts.isin(shift_params.shifts_df.handling_ts)):
                # Correct sla for non workable hours in inbound data
                df_inb_old = self.df_inb.rename(columns={'count': 'original_count',
                                                         'handling_ts': 'original_handling_ts',
                                                         'sla_ts': 'original_sla_ts'})

                self.df_inb.loc[~self.df_inb.sla_ts.isin(shift_params.shifts_df.handling_ts), 'sla_ts'] = \
                    self.df_inb.loc[~self.df_inb.sla_ts.isin(shift_params.shifts_df.handling_ts), 'sla_ts'].apply(
                        self.correct_ts_to_closest_workable_hour, args=(handlings_set,))

                df_inb_inconsistent = pd.concat([df_inb_old, self.df_inb['sla_ts']], axis=1)
                df_inb_inconsistent = df_inb_inconsistent.rename(columns={'sla_ts': 'corrected_sla_ts'})
                df_inb_inconsistent = df_inb_inconsistent[df_inb_inconsistent['original_sla_ts'] !=
                                                          df_inb_inconsistent['corrected_sla_ts']]
                # log warning because there are inconsistent slas in inbound data
                logger.warning('Corrected %s slas in non business hours in %s.\n',
                               str(len(df_inb_inconsistent)), FileNames.INBOUND_DATA)
                df_inb_inconsistent.to_csv(f"./warning/{FileNames.INCONSISTENT_INBOUND_CPT}")

            if any(~self.df_inb_initial.sla_ts.isin(shift_params.shifts_df.handling_ts)):
                # Correct sla for non workable hours in inbound initial data
                df_inb_initial_old = self.df_inb_initial.rename(columns={'count': 'original_count',
                                                                         'sla_ts': 'original_sla_ts'})

                self.df_inb_initial.loc[
                    ~self.df_inb_initial.sla_ts.isin(shift_params.shifts_df.handling_ts), 'sla_ts'] = \
                    self.df_inb_initial.loc[
                        ~self.df_inb_initial.sla_ts.isin(shift_params.shifts_df.handling_ts), 'sla_ts'].apply(
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
            if any(~self.df_out.cpt_ts.isin(shift_params.shifts_df.handling_ts)):
                # Correct cpt for non workable hours in outbound data
                df_out_old = self.df_out.rename(columns={'count': 'original_count',
                                                         'handling_ts': 'original_handling_ts',
                                                         'cpt_ts': 'original_cpt_ts'})

                self.df_out.loc[~self.df_out.cpt_ts.isin(shift_params.shifts_df.handling_ts), 'cpt_ts'] = \
                    self.df_out.loc[~self.df_out.cpt_ts.isin(shift_params.shifts_df.handling_ts), 'cpt_ts'].apply(
                        self.correct_ts_to_closest_workable_hour, args=(handlings_set,))

                df_out_inconsistent = pd.concat([df_out_old, self.df_out['cpt_ts']], axis=1)
                df_out_inconsistent = df_out_inconsistent.rename(columns={'cpt_ts': 'corrected_cpt_ts'})
                df_out_inconsistent = df_out_inconsistent[df_out_inconsistent['original_cpt_ts'] !=
                                                          df_out_inconsistent['corrected_cpt_ts']]
                # log warning
                logger.warning('Corrected %s cpts in non business hours in %s.\n',
                               len(df_out_inconsistent), FileNames.OUTBOUND_DATA)

                df_out_inconsistent.to_csv(f'./warning/{FileNames.INCONSISTENT_OUTBOUND_CPT}')

            if any(~self.df_out_initial.cpt_ts.isin(shift_params.shifts_df.handling_ts)):
                # Correct cpt for non workable hours in outbound initial data
                df_out_initial_old = self.df_out_initial.rename(columns={'count': 'original_count',
                                                                         'cpt_ts': 'original_cpt_ts'})

                self.df_out_initial.loc[
                    ~self.df_out_initial.cpt_ts.isin(shift_params.shifts_df.handling_ts), 'cpt_ts'] = \
                    self.df_out_initial.loc[
                        ~self.df_out_initial.cpt_ts.isin(shift_params.shifts_df.handling_ts), 'cpt_ts'].apply(
                        self.correct_ts_to_closest_workable_hour, args=(handlings_set,))

                df_out_initial_inconsistent = pd.concat([df_out_initial_old, self.df_out_initial['cpt_ts']], axis=1)
                df_out_initial_inconsistent = df_out_initial_inconsistent.rename(columns={'cpt_ts': 'corrected_cpt_ts'})
                df_out_initial_inconsistent = df_out_initial_inconsistent[
                    df_out_initial_inconsistent['original_cpt_ts'] !=
                    df_out_initial_inconsistent['corrected_cpt_ts']]
                logger.warning('Corrected %s cpts in non business hours in %s.\n',
                               str(len(df_out_initial_inconsistent)), FileNames.OUTBOUND_INITIAL_DATA)
                df_out_initial_inconsistent.to_csv(f"./warning/{FileNames.INCONSISTENT_OUTBOUND_INITIAL_CPT}")

        self.do_index_handlings(shift_params.df_dates)
        self.do_index_cpts(shift_params.df_dates)

        self.df_inb.sort_values('handling_idx', inplace=True)
        self.df_out.sort_values('handling_idx', inplace=True)

        if DevelopDumping.DEV or DevelopDumping.QAS or Configuration.generate_validation_files:
            self.df_inb.to_csv(f"{InputOutputPaths.BASEDIR_VAL}/{FileNames.INBOUND_CORRECTED_SLAS}")

            self.df_inb_initial.to_csv(f"{InputOutputPaths.BASEDIR_VAL}/{FileNames.INBOUND_INITIAL_CORRECTED_SLAS}")

            self.df_out.to_csv(f"{InputOutputPaths.BASEDIR_VAL}/{FileNames.OUTBOUND_CORRECTED_SLAS}")

            self.df_out_initial.to_csv(f"{InputOutputPaths.BASEDIR_VAL}/{FileNames.OUTBOUND_INITIAL_CORRECTED_SLAS}")

            inb_params.coefficients_table.to_csv(f"{InputOutputPaths.BASEDIR_VAL}/"
                                                 f"{FileNames.INBOUND_COEFS_TABLE}", index=False)

            out_params.coefficients_table.to_csv(f"{InputOutputPaths.BASEDIR_VAL}/"
                                                 f"{FileNames.OUTBOUND_COEFS_TABLE}", index=False)

            inb_params.process_workers_full_table.to_csv(f"{InputOutputPaths.BASEDIR_VAL}/"
                                                         f"{FileNames.INBOUND_WRKRS_INFO_TABLE}", index=False)

            out_params.process_workers_full_table.to_csv(f"{InputOutputPaths.BASEDIR_VAL}/"
                                                         f"{FileNames.OUTBOUND_WRKRS_INFO_TABLE}", index=False)

            shift_params.shifts_df.to_csv(f"{InputOutputPaths.BASEDIR_VAL}/"
                                          f"{FileNames.SHIFTS_TABLE_DEFINITIVE}", index=False)

        initial = self.df_out_initial.groupby(['stage', 'cpt_idx'])['count'].sum().to_dict()
        inbound_initial = self.df_inb_initial.groupby(['stage', 'sla_idx'])['count'].sum().to_dict()
        subcarriers_initial = self.df_out_initial.groupby(['stage', 'cpt_idx', 'carrier'])['count'].first().to_dict()

        inb_dh = TabulatedInboundDataHolder(self.df_inb,
                                            Process.INBOUND,
                                            shift_params.n_stages_inbound,
                                            shift_params.inbound_signature.stage_names,
                                            shift_params.df_dates[fld_names.ID].max(),
                                            initial=inbound_initial)

        out_dh = TabulatedOutboundDataHolder(self.df_out,
                                             Process.OUTBOUND,
                                             shift_params.n_stages_outbound,
                                             shift_params.outbound_signature.stage_names,
                                             shift_params.df_dates[fld_names.ID].max(),
                                             initial=initial,
                                             subcarriers_initial=subcarriers_initial)

        params = {Process.INBOUND: inb_params, Process.OUTBOUND: out_params}
        data_holders = {Process.INBOUND: inb_dh, Process.OUTBOUND: out_dh}

        if Configuration.make_rolling:
            shift_interval = Configuration.shift_interval
        else:
            # TODO: to define the last shift is delicate, sometimes the biggest index doesn't correspond to the one that
            # ends the last.
            shift_interval = shift_params.shifts.max() + 1

        # OBS: we put the params for extra hours aside because we have to think further how we are going to pass the
        # parameters at once for any number of processes
        executor = RollingExecutor(shift_interval, shift_params, data_holders,
                                   params, self.raw_eh_parameters, shift_params.df_dates)

        return executor.run()


if __name__ == "__main__":
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
        logger.info("\rMoletron Version [ver] [month] [year]\n")
        logger.info("Read all the messages in %s and in this window. Then press ENTER to close.\n",
                    FileNames.LOG_FILE)

        RollingMain().run()
        input(f"Completed!\n\nTotal execution time was {time() - __start__:.4f} seconds. Press ENTER to close. ")
    except Exception as ex:
        logger.exception("\n\n--------------------------------------------------------\n"
                         "There were errors. Read the following and the log files.\n"
                         "--------------------------------------------------------\n\n%s", ex)
        input(f"Total execution time was {time() - __start__:.4f} seconds. Press ENTER to close. ")
