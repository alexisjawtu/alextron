import logging
import pandas as pd
import kernel.data_frames_field_names as fld_names

from kernel.general_configurations import InputOutputPaths

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TestFiles:
    MAX_EPOCH: str = "max_epoch.csv"
    INBOUND_INITIAL_CORRECTED_SLAS: str = "inbound_initial_corrected_slas.csv"
    INBOUND_CORRECTED_SLAS: str = "inbound_corrected_slas.csv"
    WORKERS_INBOUND: str = "workers_inbound.csv"
    PROCESSED_INBOUND: str = "processed_inbound.csv"
    SH_NAME_TO_CONTRACT_MOD: str = "shift_contract_modality.csv"

    OUTBOUND_INITIAL_CORRECTED_SLAS: str = "outbound_initial_corrected_slas.csv"
    OUTBOUND_CORRECTED_SLAS: str = "outbound_corrected_slas.csv"
    WORKERS_OUTBOUND: str = "workers_outbound.csv"
    PROCESSED_OUTBOUND: str = "processed_outbound.csv"

    PRODUCTIVITY: str = "%s/productivity_check_%s_%s.csv"
    WORKERS_PARAMETERS: str = "workers_parameters.csv"
    PARAMETERS_TABLE_DEFINITIVE: str = "shifts_table_definitive.csv"
    KPI_TOTAL_IDLENESS: str = "kpi_total_idleness.csv"
    KPI_TOTAL_IDLENESS_WITH_EXTRAS: str = "kpi_total_idleness_with_extras.csv"

    PROCESSING: str = "%s/processing_check_%s_%s_%s.csv"
    ZERO_PROCESSING: str = "rows_with_zero_processing.csv"
    TOTAL_PROCESSING_CHECK: str = "total_processing_check.csv"

    OBJECTIVE_VALUES: str = "objective_values.csv"
    PARTIAL_ORDER_OF_SHIFTS: str = "shifts_ordered_by_week.py"


class Readers:
    BASEDIR = InputOutputPaths.BASEDIR
    BASEDIR_OUT = InputOutputPaths.BASEDIR_OUT
    BASEDIR_VAL = InputOutputPaths.BASEDIR_VAL

    @staticmethod
    def read_inbound():
        a = pd.read_csv(f"{Readers.BASEDIR_VAL}/{TestFiles.INBOUND_CORRECTED_SLAS}")
        a['handling_ts'] = pd.to_datetime(a['handling_ts'])
        a['sla_ts'] = pd.to_datetime(a['sla_ts'])
        return a

    @staticmethod
    def read_outbound():
        a = pd.read_csv(f"{Readers.BASEDIR_VAL}/{TestFiles.OUTBOUND_CORRECTED_SLAS}")
        a['handling_ts'] = pd.to_datetime(a['handling_ts'])
        a['cpt_ts'] = pd.to_datetime(a['cpt_ts'])
        return a

    @staticmethod
    def read_inbound_initial():
        a = pd.read_csv(f'{Readers.BASEDIR_VAL}/{TestFiles.INBOUND_INITIAL_CORRECTED_SLAS}')
        a['sla_ts'] = pd.to_datetime(a['sla_ts'])
        return a

    @staticmethod
    def read_outbound_initial():
        a = pd.read_csv(f'{Readers.BASEDIR_VAL}/{TestFiles.OUTBOUND_INITIAL_CORRECTED_SLAS}')
        a['cpt_ts'] = pd.to_datetime(a['cpt_ts'])
        return a

    @staticmethod
    def read_workers_inbound():
        a = pd.read_csv(f'{Readers.BASEDIR_OUT}/{TestFiles.WORKERS_INBOUND}').dropna()
        a['epoch_ts'] = pd.to_datetime(a['epoch_ts'])
        return a

    @staticmethod
    def read_workers_outbound():
        a = pd.read_csv(f'{Readers.BASEDIR_OUT}/{TestFiles.WORKERS_OUTBOUND}').dropna()
        a['epoch_ts'] = pd.to_datetime(a['epoch_ts'])
        return a

    @staticmethod
    def read_processed_inbound():
        pi = pd.read_csv(f'{Readers.BASEDIR_OUT}/{TestFiles.PROCESSED_INBOUND}')
        pi['cpt_ts'] = pd.to_datetime(pi['cpt_ts'])
        pi['epoch_ts'] = pd.to_datetime(pi['epoch_ts'])
        return pi

    @staticmethod
    def read_processed_outbound():
        po = pd.read_csv(f'{Readers.BASEDIR_OUT}/{TestFiles.PROCESSED_OUTBOUND}')
        po['cpt_ts'] = pd.to_datetime(po['cpt_ts'])
        po['epoch_ts'] = pd.to_datetime(po['epoch_ts'])
        return po


class Inbound:
    NAME = 'inbound'
    SLA = 'sla_ts'
    STAGES = ('receiving', 'checkin')
    N_STAGES = 2

    @staticmethod
    def set_parameters(params):
        Inbound.parameters = params

    @staticmethod
    def set_data():
        Inbound.data = Readers.read_inbound()
        Inbound.initial_data = Readers.read_inbound_initial()
        Inbound.workers = Readers.read_workers_inbound()
        Inbound.processed = Readers.read_processed_inbound()


class Outbound:
    NAME = 'outbound'
    SLA = 'cpt_ts'
    STAGES = ('picking', 'packing')
    N_STAGES = 2

    @staticmethod
    def set_parameters(params):
        Outbound.parameters = params

    @staticmethod
    def set_data():
        Outbound.data = Readers.read_outbound()
        Outbound.initial_data = Readers.read_outbound_initial()
        Outbound.workers = Readers.read_workers_outbound()
        Outbound.processed = Readers.read_processed_outbound()


class Hr:
    delta = pd.Timedelta(hours=1)

class OutputTest:
    def __init__(self,
                 shift_holder,
                 inbound_parameters,
                 outbound_parameters,
                 objective_value_staffing_optimization,
                 objective_value_stock_anticipation):
        self.wrkrs_params = pd.read_csv(f'{Readers.BASEDIR}/{TestFiles.WORKERS_PARAMETERS}').dropna()
        df_shift_contract_modality = pd.read_csv(f'{Readers.BASEDIR}/{TestFiles.SH_NAME_TO_CONTRACT_MOD}').dropna()
        mod_name_by_shift_name = df_shift_contract_modality.groupby(fld_names.SHIFT_NAME).agg(list)
        self.wrkrs_params = Util.explode_modality_all(mod_name_by_shift_name, self.wrkrs_params)

        self.shifts = pd.read_csv(f"{Readers.BASEDIR_VAL}/{TestFiles.PARAMETERS_TABLE_DEFINITIVE}")
        self.shift_holder = shift_holder
        self.shift_names = shift_holder.shifts_df['shift_name'].unique()
        self.shifts_for_epoch = shift_holder.shifts_df.groupby('idx')['shift_idx'].agg(list).to_dict()
        self.shift_names_by_index = shift_holder.shifts_df.groupby('shift_idx')['shift_name'].first().to_dict()
        self.inbound_parameters = inbound_parameters
        self.outbound_parameters = outbound_parameters
        self.max_epoch = pd.Timestamp(year=pd.Timestamp.today().year, month=1, day=1)
        self.objective_value_staffing_optimization = objective_value_staffing_optimization
        self.objective_value_stock_anticipation = objective_value_stock_anticipation

    def set_options(self):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.float_format', lambda x: '%.10f' % x)

        Inbound.set_parameters(self.inbound_parameters)
        Inbound.set_data()
        Outbound.set_parameters(self.outbound_parameters)
        Outbound.set_data()

    def items_workers_hourly_correspondence(self, process):
        # Check correspondence between workers in an hour and packages processed at the same hour, for each stage
        # For now, leave it as a visual check.
        for stage in range(process.N_STAGES):
            epochs = process.workers['epoch'].unique()
            records = []

            for t in epochs:
                for s in self.shifts_for_epoch[t]:
                    # do the check by hand
                    packages = process.processed[(process.processed['epoch'] == t) &
                                                 (process.processed['stage'] == stage) &
                                                 (process.processed['shift'] == s)]['processed'].sum()

                    unit_force = self.wrkrs_params[
                        (self.wrkrs_params['process'] == process.NAME) &
                        (self.wrkrs_params['stage'] == stage) &
                        (self.wrkrs_params['shift_name'] == self.shift_names_by_index[s])
                        ].groupby('condition')['work_force'].first().to_dict()

                    reps = process.workers[
                        (process.workers['epoch'] == t) &
                        (process.workers['stage'] == stage) &
                        (process.workers['shift'] == s)
                        ][['permanents_hour', 'polyvalents_hour', 'extras_for_this_shift']].iloc[0].to_dict()

                    net_work = reps['permanents_hour'] * unit_force['permanent'] + \
                               reps['polyvalents_hour'] * unit_force['polyvalent'] + \
                               reps['extras_for_this_shift'] * unit_force['polyvalent']

                    records.append({'shift_name': self.shift_names_by_index[s],
                                    'time_slot': t,
                                    'shift': s,
                                    'perm_hour': reps['permanents_hour'],
                                    'poly_hour': reps['polyvalents_hour'],
                                    'extra_hour': reps['extras_for_this_shift'],
                                    'perm_work_force': unit_force['permanent'],
                                    'poly_work_force': unit_force['polyvalent'],
                                    'processed_packages': packages,
                                    'net_work_force': net_work,
                                    'work_force_surplus': net_work - packages})

            pd.DataFrame.from_records(records).sort_values(
                'work_force_surplus'
            ).to_csv(TestFiles.PRODUCTIVITY % (Readers.BASEDIR_VAL, process.NAME, process.STAGES[stage]))

    def zero_processing(self):
        # Check if the processing output doesn't include rows with 0 processed packages
        with open(f"{Readers.BASEDIR_VAL}/{TestFiles.ZERO_PROCESSING}", "w") as z:
            z.write(f"{Inbound.NAME},{Outbound.NAME}\n"
                    f"{len(Inbound.processed[Inbound.processed['processed'] == 0])},"
                    f"{len(Outbound.processed[Outbound.processed['processed'] == 0])}")

    def total_processing(self):
        # Check globaly if the number of processed items coincide with the input for each stage j
        _input_ = []
        _processed_ = []

        for j in range(Inbound.N_STAGES):
            _processed_.append(round(
                Inbound.processed[(Inbound.processed['stage'] == j) &
                                  (Inbound.processed['cpt_ts'] <= self.max_epoch + Hr.delta)]['processed'].sum(), 2)
            )

            _input_.append(round(
                Inbound.initial_data[(Inbound.initial_data[Inbound.SLA] <= self.max_epoch + Hr.delta) &
                                     (Inbound.initial_data['stage'] <= j)]['count'].sum() +
                Inbound.data[Inbound.data[Inbound.SLA] <= self.max_epoch + Hr.delta]['count'].sum(), 2)
            )

        for j in range(Outbound.N_STAGES):
            _processed_.append(round(Outbound.processed[(Outbound.processed['stage'] == j) &
                                                        (Outbound.processed['cpt_ts'] <= self.max_epoch + Hr.delta)][
                                         'processed'].sum(), 2))

            _input_.append(
                round(Outbound.initial_data[(Outbound.initial_data[Outbound.SLA] <= self.max_epoch + Hr.delta) &
                                            (Outbound.initial_data['stage'] <= j)]['count'].sum()
                      + Outbound.data[Outbound.data[Outbound.SLA] <= self.max_epoch + Hr.delta]['count'].sum(), 2))

        indexes = dict(zip(range(len(Inbound.STAGES + Outbound.STAGES)), Inbound.STAGES + Outbound.STAGES))

        df = pd.DataFrame({'input': _input_, 'processed': _processed_})
        df['surplus'] = df['input'] - df['processed']
        if (abs(df['surplus']) >= 1).any():
            logger.warning('Some surplus for total processing is greater than one.\n')
        df.round({'surplus': 2}).rename(index=indexes).to_csv(
            f'{Readers.BASEDIR_VAL}/{TestFiles.TOTAL_PROCESSING_CHECK}'
        )

    def processing_per_sla(self, process, max_epoch):
        # Check if the number of processed items coincide with the input
        # for each stage j and for each cpt timestamp
        slas = {'initial': {j: process.initial_data[process.initial_data['stage'] == j][process.SLA].unique()
                            for j in range(process.N_STAGES)},
                'planned': {j: process.data[process.SLA].unique() for j in range(process.N_STAGES)}}
        for status, slas in iter(slas.items()):
            for j in range(process.N_STAGES):
                records = []
                for ts in slas[j]:
                    p = process.processed[(process.processed['stage'] == j) &
                                          (process.processed['cpt_ts'] == ts)]['processed'].sum()
                    i = process.data[process.data[process.SLA] == ts]['count'].sum() + \
                        process.initial_data[(process.initial_data[process.SLA] == ts) &
                                             (process.initial_data['stage'] <= j)]['count'].sum()

                    records.append({'cpt_ts': ts, 'processed': p, 'input': i, 'surplus': i - p})

                if records:
                    df = pd.DataFrame.from_records(records).round({'surplus': 4}).sort_values('surplus')
                    if ((abs(df['surplus']) >= 1) & (df['cpt_ts'] <= max_epoch)).any():
                        logger.warning(
                            'Some surplus for processing per sla for process %s in the stage %s is greater '
                            'than one.\n',
                            str(process.NAME), str(process.STAGES[j]))
                    df.to_csv(TestFiles.PROCESSING % (Readers.BASEDIR_VAL, process.NAME, status, process.STAGES[j]))

    def write_max_epoch(self):
        self.max_epoch = max(Inbound.data['handling_ts'].max(), Outbound.data['handling_ts'].max())
        with open(f"{Readers.BASEDIR_VAL}/{TestFiles.MAX_EPOCH}", "w") as m:
            m.write(str(self.max_epoch))
        return self.max_epoch

    def normalize(self, process):
        # aux function
        # Normalized means divided back by the presence rates

        def conditional_div(a, b):
            # aux function
            return a / b if b else 0

        table = process.workers
        #TODO verificar, no estoy segura que modifique que impacto para tener que cambiar esto
        modality = self.wrkrs_params['contract_modality'].iloc[0]
        table['permanents_hour'] = table.apply(lambda t: conditional_div(t['permanents_hour'],
                                                                         process.parameters.presence_rate(modality,
                                                                             t['epoch'], t['shift'])), axis=1)
        table['polyvalents_hour'] = table.apply(lambda t: conditional_div(t['polyvalents_hour'],
                                                                          process.parameters.presence_rate(modality,
                                                                              t['epoch'], t['shift'])), axis=1)
        return table

    def test_balance(self):
        # This intends to check two versions of the important concept of 'idleness' in an operation.
        workers_group = pd.concat([self.normalize(Inbound), self.normalize(Outbound)])
        workers_group = workers_group.groupby(['process', 'day', 'shift_name', 'epoch_ts', 'epoch'])

        workers_no_extras = workers_group[['permanents_hour', 'polyvalents_hour']].sum().reset_index()
        workers_with_extras = workers_group[['permanents_hour', 'polyvalents_hour',
                                             'extras_for_this_shift']].sum().reset_index()

        by_epoch = workers_no_extras.groupby(['shift_name', 'epoch']).sum().reset_index()
        by_epoch_xtr = workers_with_extras.groupby(['shift_name', 'epoch']).sum().reset_index()

        by_epoch['sum'] = by_epoch[['permanents_hour', 'polyvalents_hour']].sum(axis=1)
        by_epoch_xtr['sum'] = by_epoch_xtr[['permanents_hour', 'polyvalents_hour', 'extras_for_this_shift']].sum(axis=1)

        max_over_epochs_in_shift_name = by_epoch.groupby('shift_name')['sum'].max()
        max_over_epochs_in_shift_name_xtr = by_epoch_xtr.groupby('shift_name')['sum'].max()

        idleness = {}
        idleness_xtr = {}

        for name in self.shift_names:
            n_slots = Outbound.workers[(Outbound.workers['epoch_ts'] <= self.max_epoch) &
                                       (Outbound.workers['shift_name'] == name)]['epoch'].nunique()

            numerator = workers_no_extras[workers_no_extras['shift_name'] == name][
                ['permanents_hour', 'polyvalents_hour']].sum().sum()

            numerator_xtr = workers_with_extras[workers_with_extras['shift_name'] == name][
                ['permanents_hour', 'polyvalents_hour', 'extras_for_this_shift']].sum().sum()

            denominator = n_slots * max_over_epochs_in_shift_name[name]
            denominator_xtr = n_slots * max_over_epochs_in_shift_name_xtr[name]

            if denominator:
                idleness[name] = 1 - numerator / denominator

            if denominator_xtr:
                idleness_xtr[name] = 1 - numerator_xtr / denominator_xtr

        str_warning = "Some idleness is greater than 0.3 for the stages "
        idle = 0

        str_info = "Total idleness table\n"
        str_info += "{:<20} {:<20}\n".format("Stage", "Total idleness")

        for k, v in idleness.items():
            str_info += "{:<20} {:<20}\n".format(k, v)

            if abs(v) > 0.3:
                str_warning += f"{k},\n"
                idle += 1

        str_info += '\n'

        str_warning_xtr = "Some idleness with extras is greater than 0.3 for the stages "
        idle_xtr = 0

        str_info += "Total idleness with extras table\n"
        str_info += "{:<20} {:<20}\n".format("Stage", "Total idleness")
        for k, v in idleness_xtr.items():
            str_info += ("{:<20} {:<20}\n".format(k, v))
            if abs(v) > 0.3:
                str_warning_xtr += f"{k},\n"

        str_warning += 2 * "\b"
        str_warning_xtr += 2 * "\b"

        pd.Series(idleness, name="Total idleness").to_csv(f"{Readers.BASEDIR_VAL}/{TestFiles.KPI_TOTAL_IDLENESS}")
        pd.Series(idleness_xtr, name="Total idleness with extras") \
            .to_csv(f"{Readers.BASEDIR_VAL}/{TestFiles.KPI_TOTAL_IDLENESS_WITH_EXTRAS}")

        logger.info(str_info)

        if idle:
            logger.warning(str_warning)

        if idle_xtr:
            logger.warning(str_warning_xtr)

    def write_objective_value(self):
        with open(f"{Readers.BASEDIR_VAL}/{TestFiles.OBJECTIVE_VALUES}", "w") as m:
            m.write("staffing_optimization_value,stock_anticipation_value\n")
            m.write("{:.10f},{:.10f}".format(self.objective_value_staffing_optimization,
                                             self.objective_value_stock_anticipation))

    def generate_check_tables(self):
        self.set_options()
        max_epoch = self.write_max_epoch()

        # TODO: the following method doesn't have the scheduled shift_names, in case 
        # there is any in the present scheduling instance, ahd therefore throws a KeyError Exception
        # self.items_workers_hourly_correspondence(Inbound)
        # self.items_workers_hourly_correspondence(Outbound)

        self.zero_processing()
        self.total_processing()

        self.processing_per_sla(Inbound, max_epoch)
        self.processing_per_sla(Outbound, max_epoch)

        self.write_objective_value()

        self.test_balance()

class Util:
    TODO: take this out of here. We should have one explode in the library and call it
    def explode_modality_all(mod_name_by_shift_name, df):
        if not df.empty:
            df_all = df.loc[
                (df[fld_names.HIRING_MODALITY] == fld_names.ALL_MODALITY)]
            df.drop(df_all.index, inplace=True)
            if not df_all.empty:
                df_all = df_all.drop(fld_names.HIRING_MODALITY, axis='columns')
                df_all = df_all.merge(mod_name_by_shift_name, how='inner', on=fld_names.SHIFT_NAME)
                df_all = df_all.explode(fld_names.HIRING_MODALITY)
                df = df.append(df_all)
        return df
