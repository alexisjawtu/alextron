import logging
import pandas as pd

from pathlib import Path
from typing import Dict, Tuple, List
from dateutil.parser import ParserError

import kernel.data_frames_field_names as fld_names

from kernel.general_configurations import InputOutputPaths, FileNames

logger = logging.getLogger(__name__)


def path_string(filename: str) -> str:
    return f"{InputOutputPaths.BASEDIR}/{filename}"


def auxiliary_standard_read(filename) -> pd.DataFrame:
    try:
        df = pd.read_csv(path_string(filename)).dropna()

    except FileNotFoundError as fnf_error:
        df = pd.DataFrame()
        log_error(fnf_error, filename)

    return df


def auxiliary_standard_read_for_scheduling(filename) -> pd.DataFrame:
    """ This reader is just a way of checking whether the optional 
    inputs for scheduling were included or not """
    try:
        df = pd.read_csv(path_string(filename)).dropna()
        df[fld_names.VAL_START] = pd.to_datetime(df[fld_names.VAL_START])
        df[fld_names.VAL_END] = pd.to_datetime(df[fld_names.VAL_END])

    except FileNotFoundError:
        df = pd.DataFrame()
    except KeyError:
        logger.error("Missing %s or %s fields in file %s." % (fld_names.VAL_START, fld_names.VAL_END, filename))

    return df


def read_data() -> Tuple[pd.DataFrame]:
    df_out = auxiliary_standard_read(FileNames.OUTBOUND_DATA)

    # TODO: the field 'stage' remains '1.0' instead of '1' in the dfs
    try:
        df_out['cpt_ts'] = pd.to_datetime(df_out['cpt_ts'])
    except ParserError as ps_error:
        log_parser_error(ps_error, 'cpt_ts', FileNames.OUTBOUND_DATA)

    try:
        df_out['handling_ts'] = pd.to_datetime(df_out['handling_ts'])
    except ParserError as ps_error:
        log_parser_error(ps_error, 'handling_ts', FileNames.OUTBOUND_DATA)

    df_out.columns = map(lambda s: s.lower(), df_out.columns)

    df_out_initial = auxiliary_standard_read(FileNames.OUTBOUND_INITIAL_DATA)
    check_stage_valid(df_out_initial, 'stage', FileNames.OUTBOUND_INITIAL_DATA)

    df_out_initial.columns = map(lambda s: s.lower(), df_out_initial.columns)

    try:
        df_out_initial['cpt_ts'] = pd.to_datetime(df_out_initial['cpt_ts'])
    except ParserError as ps_error:
        log_parser_error(ps_error, 'cpt_ts', FileNames.OUTBOUND_INITIAL_DATA)

    try:
        df_out['count'] = df_out['count'].astype(int)
        df_out_initial['count'] = df_out_initial['count'].astype(int)
    except ParserError as ps_error:
        log_parser_error(ps_error, 'count', FileNames.OUTBOUND_INITIAL_DATA)

    try:
        df_out['carrier'] = df_out['carrier'].apply(lambda s: s.replace(' ', ''))
        df_out_initial['carrier'] = df_out_initial['carrier'].apply(lambda s: s.replace(' ', ''))
    except KeyError:
        logger.warning('Field "carrier" not found in %s or in %s. '
                       'However, this instance will keep running.\n',
                       FileNames.OUTBOUND_DATA, FileNames.OUTBOUND_INITIAL_DATA)

    df_inb = auxiliary_standard_read(FileNames.INBOUND_DATA)

    try:
        df_inb['sla_ts'] = pd.to_datetime(df_inb['sla_ts'])
    except ParserError as ps_error:
        log_parser_error(ps_error, 'sla_ts', FileNames.INBOUND_DATA)

    try:
        df_inb['handling_ts'] = pd.to_datetime(df_inb['handling_ts'])
    except ParserError as ps_error:
        log_parser_error(ps_error, 'handling_ts', FileNames.INBOUND_DATA)
    df_inb.columns = map(lambda s: s.lower(), df_inb.columns)

    df_inb_initial = auxiliary_standard_read(FileNames.INBOUND_INITIAL_DATA)

    check_stage_valid(df_inb_initial, 'stage', FileNames.INBOUND_INITIAL_DATA)

    df_inb_initial.columns = map(lambda s: s.lower(), df_inb_initial.columns)

    try:
        df_inb_initial['sla_ts'] = pd.to_datetime(df_inb_initial['sla_ts'])
    except ParserError as ps_error:
        log_parser_error(ps_error, 'sla_ts', FileNames.INBOUND_INITIAL_DATA)

    try:
        df_inb['count'] = df_inb['count'].astype(int)
        df_inb_initial['count'] = df_inb_initial['count'].astype(int)
    except ParserError as ps_error:
        log_parser_error(ps_error, 'count', FileNames.INBOUND_INITIAL_DATA)

    return df_inb, df_out, df_inb_initial, df_out_initial


def log_error(error: FileNotFoundError, file: str) -> None:
    logger.error(error)
    input(f"A fatal error occurred! File {file} was not found. Press ENTER to close.")


def log_parser_error(error: ParserError, column: str, file: str) -> None:
    logger.error(error)
    input(f"A parser error has occurred in column {column} of file {file}. Press ENTER to close.")


def check_day_names(df: pd.DataFrame, column: str, file: str) -> None:
    if (~df[column].isin(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])).any():
        logger.error(f'An error has occurred parsing a day in column {column} of file {file}.')
        input("Press ENTER to close.")
        raise Exception(f'An error has occurred parsing a day in column {column} of file {file}.')


def check_stage_valid(df: pd.DataFrame, column: str, file: str) -> None:
    if ((df[column] != 0) & (df[column] != 1)).any():
        logger.error(f'An error has occurred parsing a stage value in column {column} of file {file}.')
        input("Press ENTER to close.")
        raise Exception(f'An error has occurred parsing a stage value in column {column} of file {file}.')


def check_hour_valid(df: pd.DataFrame, column: str, file: str) -> None:
    if ~((df[column] >= 0) & (df[column] <= 23)).any():
        logger.error(f'An error has occurred parsing an hour value in column {column} of file {file}.')
        input("Press ENTER to close.")
        raise Exception(f'An error has occurred parsing an hour value in column {column} of file {file}.')


def read_shifts() -> pd.DataFrame:
    """
    Docstring: save the hours and minutes and make an integer version of 
    "start" and "end" columns, to achieve backwards compatibility with the
    algorithms to build shifts 

    We end up with:

        df = df[["day_name",
                 "shift_name",
                 "start",
                 "start_minute",
                 "end",
                 "end_hour",
                 "end_minute"]]
    """
    df = auxiliary_standard_read(FileNames.SHIFTS_PARAMS)
    check_day_names(df, "day_name", FileNames.SHIFTS_PARAMS)

    anihilated = (df["start"] == "-1") | (df["end"] == "-1")

    df[["start", "start_minute"]] = df["start"].astype(str).str.split(":", expand=True).fillna(-1).astype(int)
    df[["end_hour", "end_minute"]] = df["end"].astype(str).str.split(":", expand=True).fillna(-1).astype(int)

    # "end_hour" is the original input hour and "end" is the standard integer end as always.
    df["end"] = df["end_hour"]
    df.loc[df["end_minute"] > 0, "end"] = (df.loc[df["end_minute"] > 0, "end"] + 1).mod(24)

    return df


def read_shifts_scheduling() -> pd.DataFrame:
    """ 
    Docstring: save the hours and minutes and make an integer version of 
    "start" and "end" columns, to achieve backwards compatibility with the
    algorithms to build shifts

    We end up with these columns:

        ret = ret[["validity_start",
                   "validity_end",
                   "shift_name",
                   "start",
                   "start_minute",
                   "end",
                   "end_hour",
                   "end_minute"]]
    """
    path_str = f"{InputOutputPaths.BASEDIR}/{FileNames.SHIFTS_PARAMS_SCHEDULING}"
    path = Path(path_str)
    ret = pd.DataFrame()
    if path.is_file():
        try:
            ret = pd.read_csv(path_str).dropna()
        except FileNotFoundError as fnf_error:
            log_error(fnf_error, FileNames.SHIFTS_PARAMS_SCHEDULING)

        anihilated = (ret["start"] == "-1") | (ret["end"] == "-1")

        ret.validity_start = pd.to_datetime(ret.validity_start)
        ret.validity_end = pd.to_datetime(ret.validity_end)

        ret[["start", "start_minute"]] = ret["start"].astype(str).str.split(":", expand=True).fillna(-1).astype(int)
        ret[["end_hour", "end_minute"]] = ret["end"].astype(str).str.split(":", expand=True).fillna(-1).astype(int)
        
        ret["end"] = ret["end_hour"]
        ret.loc[ret["end_minute"] > 0, "end"] = (ret.loc[ret["end_minute"] > 0, "end"] + 1).mod(24)

    return ret


raw_local = read_shifts()
raw_sched_local = read_shifts_scheduling()
if len(raw_sched_local):
    shift_names = pd.concat([raw_local['shift_name'], raw_sched_local['shift_name']]).unique().tolist()
else:
    shift_names = raw_local['shift_name'].unique().tolist()
imported_shift_names_and_indices: List = dict(zip(shift_names, range(len(shift_names))))


def read_workers_parameters() -> pd.DataFrame:
    return auxiliary_standard_read(FileNames.WORKERS_PARAMETERS)


def read_backlog_bounds() -> pd.DataFrame:
    df = auxiliary_standard_read(FileNames.BACKLOG_BOUNDS)
    check_stage_valid(df, 'stage', FileNames.BACKLOG_BOUNDS)
    return df


def read_presences() -> pd.DataFrame:
    df = auxiliary_standard_read(FileNames.PRESENCES)
    check_hour_valid(df, 'hour', FileNames.PRESENCES)
    return df


def read_presences_scheduling() -> Tuple[pd.DataFrame]:
    # TODO: unify the reading with auxiliary_standard_read_for_scheduling()
    path_str = f'{InputOutputPaths.BASEDIR}/{FileNames.PRESENCES_SCHEDULING}'
    path = Path(path_str)
    ret = pd.DataFrame(), pd.DataFrame()
    if path.is_file():  # Try to read if it exists
        try:
            table = pd.read_csv(path_str).dropna()
        except FileNotFoundError as fnf_error:
            log_error(fnf_error, FileNames.PRESENCES_SCHEDULING)
        table.hour = pd.to_datetime(table.hour)
        # TODO: we can do this in a lot of places, just in case
        table.process = table.process.apply(lambda s: s.lower())
        ret = table[table.process == 'inbound'], table[table.process == 'outbound']
    return ret


def read_workers_parameters_scheduling() -> Tuple[pd.DataFrame]:
    path_str = f'{InputOutputPaths.BASEDIR}/{FileNames.WORKERS_PARAMETERS_SCHEDULING}'
    path = Path(path_str)
    ret = pd.DataFrame(), pd.DataFrame()
    if path.is_file():
        try:
            table = pd.read_csv(path_str).dropna()
        except FileNotFoundError as fnf_error:
            log_error(fnf_error, FileNames.WORKERS_PARAMETERS_SCHEDULING)
        check_stage_valid(table, 'stage', FileNames.WORKERS_PARAMETERS_SCHEDULING)
        table.validity_start = pd.to_datetime(table.validity_start)
        table.validity_end = pd.to_datetime(table.validity_end)
        ret = table[table.process == 'inbound'], table[table.process == 'outbound']
    return ret


def read_transfers():
    df_transfers = auxiliary_standard_read(FileNames.TRANSFERS)
    gb = df_transfers.groupby(fld_names.HIRING_MODALITY)

    dc_shifts_from = {}
    dc_shifts_to = {}

    for gkey in gb.groups:
        # TODO: here in each group we should check for repeated records in the input
        group_df = gb.get_group(gkey).drop(columns=fld_names.HIRING_MODALITY)

        dc_shifts_from[gkey] = group_df.groupby(fld_names.SHIFT_FROM)[fld_names.SHIFT_TO].unique().to_dict()
        dc_shifts_to[gkey] = group_df.groupby(fld_names.SHIFT_TO)[fld_names.SHIFT_FROM].unique().to_dict()

    return df_transfers, dc_shifts_from, dc_shifts_to


def read_workers_costs():
    return auxiliary_standard_read(FileNames.WORKERS_COSTS)


def read_workers_initial(mod_name_by_shift_type: dict) -> Tuple[Dict]:
    """
        Added dict to match modality ALL with shifts modalities
    """

    def make_dict_aux(df: pd.DataFrame) -> Dict:
        """
            We want the following to be returned:

                {(hiring_modality, shift_type, stage): initial_qtty}
                {(hiring_modality, shift_type, stage): initial_qtty}

        Example:
            {
                ('MeLi_perm', 'AFTERNOON0', 0): 50,
                ('DHL_temp', 'AFTERNOON0', 1): 80,
                ('MeLi_perm', 'MORNING0', 1): 37
            }
        """
        return df.groupby(
            [fld_names.HIRING_MODALITY, fld_names.SHIFT_TYPE, fld_names.STAGE]
        )[fld_names.INITIAL_WORKERS].first().to_dict()

    try:
        df_wrkrs_ini = auxiliary_standard_read(FileNames.WORKERS_INITIAL)
        df_wrkrs_ini = explode_modality_all(mod_name_by_shift_type, df_wrkrs_ini)

        wici = make_dict_aux(df_wrkrs_ini[df_wrkrs_ini[fld_names.PROCESS] == fld_names.INBOUND_NAME])
        wico = make_dict_aux(df_wrkrs_ini[df_wrkrs_ini[fld_names.PROCESS] == fld_names.OUTBOUND_NAME])

    except FileNotFoundError as fnf_error:
        wici = {}
        wico = {}

    return wici, wico


def read_shift_contract_modality() -> Tuple[Dict[str, List[str]]]:
    """
        Example:

          shift_name contract_mod
        0     AFT0W1          M_P
        1     AFT0W1          M_D
        2   AFT0W1_T          M_T

        dc1 = {"AFT0W1": ["M_P", "M_D"], "AFT0W1_T": ["M_T"]}

        dc2 = {"M_P": ["AFT0W1"], "M_T": ["AFT0W1_T"], "M_D": ["AFT0W1"]}
    """

    try:
        df = auxiliary_standard_read(FileNames.SH_NAME_TO_CONTRACT_MOD)
        dc1 = df.groupby(fld_names.SHIFT_NAME)[fld_names.HIRING_MODALITY].unique().to_dict()
        dc2 = df.groupby(fld_names.HIRING_MODALITY)[fld_names.SHIFT_NAME].unique().to_dict()

    except FileNotFoundError as fnf_error:
        log_error(fnf_error, FileNames.SH_NAME_TO_CONTRACT_MOD)
        dc1 = {}
        dc2 = {}

    return dc1, dc2


def read_contract_modality_type() -> Dict[str, List[str]]:
    """
        The output is

            {"Diarista": ["MELI_Diarista"],
             "Perm": ["MELI_Perm", "DHL_Perm"],
             "Temporal": ["MELI_Temporal"]}
    """

    try:
        dc1 = auxiliary_standard_read(FileNames.CONTRACT_SUBCLASSES).groupby(
            fld_names.HIRING_TYPE).agg(list).to_dict()
        dc1 = dc1[fld_names.HIRING_MODALITY]

        dc2 = auxiliary_standard_read(FileNames.CONTRACT_SUBCLASSES).groupby(
            fld_names.HIRING_MODALITY).agg(list).to_dict()
        dc2 = dc2[fld_names.HIRING_TYPE]

    except FileNotFoundError as fnf_error:
        log_error(fnf_error, FileNames.CONTRACT_SUBCLASSES)
        dc1 = {}
        dc2 = {}

    return dc1, dc2


def read_polyvalence_parameters() -> pd.DataFrame:
    aux = auxiliary_standard_read(FileNames.POLYV_PARAMS)
    try:
        aux[fld_names.PROC_ORIG] = aux[fld_names.PROC_ORIG].str.lower()
        aux[fld_names.PROC_DEST] = aux[fld_names.PROC_DEST].str.lower()
    except KeyError:
        logger.error(f"Fields {fld_names.PROC_ORIG} or {fld_names.PROC_DEST} "
                     f"not found in file '{FileNames.POLYV_PARAMS}'.\n")
        input("Press ENTER to close.")
        exit()
    return aux


def read_absences() -> pd.DataFrame:
    return auxiliary_standard_read(FileNames.ABSENT_TABLE)


def read_absences_scheduling() -> pd.DataFrame:
    return auxiliary_standard_read_for_scheduling(FileNames.ABSENT_TABLE_SCHEDULING)


def explode_modality_all(mod_name_by_shift_type, df):
    # Function that replaces the modality "ALL" with the modalities associated with that shift type
    # Requires field names HIRING_MODALITY and SHIFT_NAME in the both tables
    if not df.empty:
        df_all = df.loc[
            ((df[fld_names.HIRING_MODALITY]).str.upper() == fld_names.ALL_MODALITY)]
        df.drop(df_all.index, inplace=True)
        if not df_all.empty:
            df_all = df_all.drop(fld_names.HIRING_MODALITY, axis='columns')
            df_all = df_all.merge(mod_name_by_shift_type, how='inner', on=fld_names.SHIFT_TYPE)
            df_all = df_all.explode(fld_names.HIRING_MODALITY)
            df = df.append(df_all)
    return df


def read_extra_hours_parameters() -> pd.DataFrame:
    return auxiliary_standard_read(FileNames.EXTRA_HOURS_PARAMETERS)


class Input:
    def __init__(self):
        self.raw_shifts: pd.DataFrame = read_shifts()
        self.raw_shifts_scheduled: pd.DataFrame = read_shifts_scheduling()
        self.backlog_bounds: pd.DataFrame = read_backlog_bounds()
        self.df_shift_contract_modality: pd.DataFrame = auxiliary_standard_read(FileNames.SH_NAME_TO_CONTRACT_MOD)
        self.polyvalents_parameters: pd.DataFrame = read_polyvalence_parameters()
        self.df_presences: pd.DataFrame = read_presences()
        self.df_presences_sched_inbound, self.df_presences_sched_outbound = read_presences_scheduling()
        self.df_absences: pd.DataFrame = read_absences()
        self.df_absences_sched: pd.DataFrame = read_absences_scheduling()
        self.df_work_forces_sched_inbound, self.df_work_forces_sched_outbound = read_workers_parameters_scheduling()
        self.df_work_forces: pd.DataFrame = read_workers_parameters()
        self.modalities: pd.DataFrame = auxiliary_standard_read(FileNames.CONTRACT_SUBCLASSES)
