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
    input(f"Fatal error. File {file} not found. Press ENTER to close.")


def log_parser_error(error: ParserError, column: str, file: str) -> None:
    logger.error(error)
    input(f"Error parsing column {column} of file {file}. Press ENTER to close.")


def check_day_names(df: pd.DataFrame, column: str, file: str) -> None:
    if (~df[column].isin(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])).any():
        logger.error(f'Error parsing day name in column {column} of file {file}.')
        input("Press ENTER to close.")
        raise Exception(f'Error parsing day name in column {column} of file {file}.')


def check_stage_valid(df: pd.DataFrame, column: str, file: str) -> None:
    if ((df[column] != 0) & (df[column] != 1)).any():
        logger.error(f'Error parsing stage value in column {column} of file {file}.')
        input("Press ENTER to close.")
        raise Exception(f'Error parsing stage value in column {column} of file {file}.')


def check_hour_valid(df: pd.DataFrame, column: str, file: str) -> None:
    if ~((df[column] >= 0) & (df[column] <= 23)).any():
        logger.error(f'Error parsing hour value in column {column} of file {file}.')
        input("Press ENTER to close.")
        raise Exception(f'Error parsing hour value in column {column} of file {file}.')


def read_shifts() -> pd.DataFrame:
    df = auxiliary_standard_read(FileNames.SHIFTS_PARAMS)
    check_day_names(df, 'day_name', FileNames.SHIFTS_PARAMS)
    return df


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


def read_shifts_scheduling() -> pd.DataFrame:
    path_str = f'{InputOutputPaths.BASEDIR}/{FileNames.SHIFTS_PARAMS_SCHEDULING}'
    path = Path(path_str)
    ret = pd.DataFrame()
    if path.is_file():
        try:
            ret = pd.read_csv(path_str).dropna()
        except FileNotFoundError as fnf_error:
            log_error(fnf_error, FileNames.SHIFTS_PARAMS_SCHEDULING)
        ret.validity_start = pd.to_datetime(ret.validity_start)
        ret.validity_end = pd.to_datetime(ret.validity_end)
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

    except FileNotFoundError:
        df = pd.DataFrame()

    return df


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


def read_workers_initial() -> Tuple[Dict]:
    def make_dict_aux(df: pd.DataFrame) -> Dict:
        """
        We want the following to be returned:

                {(hiring_modality, shift_type, stage): initial_qtty}

        Example:
            {
                ('Internal_Permanent', 'AFTERNOON0', 0): 50,
                ('DHL_Temporary', 'AFTERNOON0', 1): 80,
                ('Internal_Permanent', 'MORNING0', 1): 37
            }
        """
        return df.groupby(
                [fld_names.HIRING_MODALITY, fld_names.SHIFT_TYPE, fld_names.STAGE]
               )[fld_names.INITIAL_WORKERS].first().to_dict()

    try:
        df_wrkrs_ini = auxiliary_standard_read(FileNames.WORKERS_INITIAL)

        wici = make_dict_aux(df_wrkrs_ini[df_wrkrs_ini[fld_names.PROCESS]==fld_names.INBOUND_NAME])
        wico = make_dict_aux(df_wrkrs_ini[df_wrkrs_ini[fld_names.PROCESS]==fld_names.OUTBOUND_NAME])

    except FileNotFoundError as fnf_error:
        wici = {} 
        wico = {}

    return wici, wico


def read_shift_contract_modality() -> Tuple[Dict[str, List[str]]]:
    """
        Example:

          shift_name contract_mod
        0     AFT0W1          Internal_Permanent
        1     AFT0W1          Internal_Day_Laborer
        2   AFT0W1_T          Internal_Temporal

        dc1 = {'AFT0W1': ['Internal_Permanent', 'Internal_Day_Laborer'], 'AFT0W1_T': ['Internal_Temporal']}

        dc2 = {"Internal_Permanent": ["AFT0W1"], "Internal_Temporal": ["AFT0W1_T"], "Internal_Day_Laborer": ["AFT0W1"]}
    """

    try:
        dc = auxiliary_standard_read(FileNames.SH_NAME_TO_CONTRACT_MOD)
        dc1 = dc.groupby(fld_names.SHIFT_NAME).agg(list).to_dict()
        dc1 = dc1[fld_names.HIRING_MODALITY]

        dc2 = dc.groupby(fld_names.HIRING_MODALITY).agg(list).to_dict()
        dc2 = dc2[fld_names.SHIFT_NAME]

    except FileNotFoundError as fnf_error:
        log_error(fnf_error, FileNames.SH_NAME_TO_CONTRACT_MOD)
        dc1 = {}
        dc2 = {}

    return dc1, dc2


def read_contract_modality_type() -> Dict[str, List[str]]:
    """
        The output is

            {'Day_Laborer': ['Internal_Day_Laborer'],
             'Permanent': ['Internal_Permanent', 'DHL_Permanent'],
             'Temporary': ['Internal_Temporary']}
    """

    try:
        dc = auxiliary_standard_read(FileNames.CONTRACT_SUBCLASSES).groupby(
            fld_names.HIRING_TYPE).agg(list).to_dict()
        dc = dc[fld_names.HIRING_MODALITY]

    except FileNotFoundError as fnf_error:
        log_error(fnf_error, FileNames.CONTRACT_SUBCLASSES)
        dc = {}

    return dc


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
