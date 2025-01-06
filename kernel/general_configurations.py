import logging
import pandas as pd

from dataclasses import dataclass
from typing import Any
import kernel.data_frames_field_names as fld_names

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def cast_to_bool(s: Any) -> bool:
    # If we put numbers and booleans in configuration.csv, Pandas will leave all of them as plain strings.
    s = str(s).lower()
    b = None

    if s == "true":
        b = True
    elif s == "false":
        b = False
    else:
        logger.error("Please review the boolean values in " + FileNames.CONFIG_FILE)
        input("Press ENTER to close.")
        exit()

    return b


def cast_to_bool(s: Any) -> bool:
    # If we put numbers and booleans in configuration.csv, Pandas will leave all of them as plain strings.
    s = str(s).lower()
    b = None

    if s == "true":
        b = True
    elif s == "false":
        b = False
    else:
        logger.error("Please review the boolean values in " + FileNames.CONFIG_FILE)
        input("Press ENTER to close.")
        exit()

    return b


@dataclass
class FileNames:
    INPUT_FOLDER: str = "./input/"
    OUTPUT_FOLDER: str = "./output/"
    VALIDATION_FOLDER: str = "./validation/"
    WARNING_FOLDER: str = "./warning/"

    LOG_FILE: str = "moletron.log"
    CONFIG_FILE: str = "configuration.csv"
    # problem files
    CPX_FILE: str = "staffing"

    RAW_SOL: str = "raw_solution_%d.xml"
    VAR_NAMES: str = "variable_names_%d.txt"
    CONSTR_NAMES: str = "constraint_names_%d.txt"

    TRANSFERS: str = "transfers.csv"
    SH_NAME_TO_CONTRACT_MOD: str = "shift_contract_modality.csv"
    CONTRACT_SUBCLASSES: str = "contract_modality_type.csv"

    SHIFTS_PARAMS: str = "shifts_parameters.csv"
    SHIFTS_PARAMS_SCHEDULING: str = "shifts_parameters_scheduling.csv"

    WORKERS_COSTS: str = "workers_costs.csv"
    WORKERS_INITIAL: str = "workers_initial.csv"
    WORKERS_PARAMETERS: str = "workers_parameters.csv"
    WORKERS_PARAMETERS_SCHEDULING: str = "workers_parameters_scheduling.csv"
    POLYV_PARAMS: str = "polyvalence_parameters.csv"
    ABSENT_TABLE: str = "absences.csv"
    ABSENT_TABLE_SCHEDULING: str = "absences_scheduling.csv"
    
    PRESENCES: str = "presences.csv"
    PRESENCES_SCHEDULING: str = "presences_scheduling.csv"
    
    BACKLOG_BOUNDS: str = "backlog_bounds.csv"
    BACKLOG_BOUNDS_SCHEDULING: str = "backlog_bounds_scheduling.csv"

    INBOUND_DATA: str = "inbound.csv"
    INBOUND_INITIAL_DATA: str = "inbound_initial.csv"
    INCONSISTENT_INBOUND_INITIAL_CPT: str = "inconsistent_inbound_initial_sla.csv"
    INCONSISTENT_INBOUND_CPT: str = "inconsistent_inbound_sla.csv"
    INBOUND_INITIAL_CORRECTED_SLAS: str = "inbound_initial_corrected_slas.csv"
    INBOUND_CORRECTED_SLAS: str = "inbound_corrected_slas.csv"
    
    INBOUND_COEFS_TABLE: str = "inbound_check_coefficients_table_added.csv"
    INBOUND_WRKRS_INFO_TABLE: str = "inbound_check_process_workers_full_table.csv"

    OUTBOUND_INITIAL_DATA: str = "outbound_initial.csv"
    OUTBOUND_DATA: str = "outbound.csv"
    INCONSISTENT_OUTBOUND_INITIAL_CPT: str = "inconsistent_outbound_initial_cpt.csv"
    INCONSISTENT_OUTBOUND_CPT: str = "inconsistent_outbound_cpt.csv"
    OUTBOUND_CORRECTED_SLAS: str = "outbound_corrected_slas.csv"
    OUTBOUND_INITIAL_CORRECTED_SLAS: str = "outbound_initial_corrected_slas.csv"

    OUTBOUND_COEFS_TABLE: str = "outbound_check_coefficients_table_added.csv"
    OUTBOUND_WRKRS_INFO_TABLE: str = "outbound_check_process_workers_full_table.csv"

    PARAMS_TABLE_DEFINITIVE: str = "parameters_table_definitive.csv"
    PARTIAL_ORDER_OF_SHIFTS: str = "shifts_ordered_by_week.py"

    ITEMS_OUTPUT: str = "%s/processed_%s.csv"
    WRKRS_OUTPUT: str = "%s/workers_%s.csv"
    POLYS_OUTPUT: str = "%s/polyvalents_moves.csv"
    HOURLY_WRKRS_OUTPUT: str = "%s/hourly_workers_%s.csv"

    RESCHEDULE_RECORDS: str = "reschedule_records.csv"
    DF_WIPED_OUT: str = "df_wiped_out.csv"
    ACTIVE_RECORDS: str = "active_records.csv"
    NEW_DF_RECS: str = "new_df_recs.csv"

    TOTAL_WORKERS_GROUP_BY_DAY_SHIFT_STAGE: str = "total_workers_{}_by_day_shift_stage.csv"
    TOTAL_WORKERS_GROUP_BY_DAY_SHIFT: str = "total_workers_{}_by_day_shift.csv"
    TOTAL_WORKERS_GROUP_BY_SHIFT: str = "total_workers_{}_by_shift.csv"


@dataclass
class DevelopDumping:
    SUBDIR: str = ""  # "3_modality_1_with_absences"  # "3_Modality_New" #"3_Miau_1_modality" # "3_Costo"
    DEV: bool = False
    QAS: bool = False
    MAKE_OUTPUT: bool = True


@dataclass
class InputOutputPaths:
    dd: DevelopDumping = DevelopDumping()

    BASEDIR: str = FileNames.INPUT_FOLDER + dd.SUBDIR * dd.DEV
    BASEDIR_OUT: str = FileNames.OUTPUT_FOLDER + dd.SUBDIR * dd.DEV
    BASEDIR_VAL: str = FileNames.VALIDATION_FOLDER + dd.SUBDIR * dd.DEV
    BASEDIR_WARN: str = FileNames.WARNING_FOLDER + dd.SUBDIR * dd.DEV


@dataclass
class Configuration:
    path_str: str = f"{InputOutputPaths.BASEDIR}/{FileNames.CONFIG_FILE}"

    try:
        dc_settings = pd.read_csv(path_str).dropna().groupby("option")["value"].first().to_dict()
    except FileNotFoundError:
        dc_settings = {}
    except KeyError:
        dc_settings = {}
        logger.error(f"Fields of file '{FileNames.CONFIG_FILE}' must be 'option' and 'value'.\n")
        input("Press ENTER to close.")
        exit()

    # inner confs
    make_rolling = False
    shift_interval = 1
    display = 2
    barrier_display = 2
    params_display = 1
    mip_gap = .009
    time_limit_seconds = 60*60

    # user defined confs
    anticipate_backlog: bool = cast_to_bool(dc_settings.get(fld_names.ANTICIPATE_BACKLOG, "true"))
    fix_cpts_outbound: bool = cast_to_bool(dc_settings.get(fld_names.FIX_CPTS_OUTBOUND, "true"))
    fix_slas_inbound: bool = cast_to_bool(dc_settings.get(fld_names.FIX_SLAS_INBOUND, "true"))
    generate_validation_files: bool = cast_to_bool(dc_settings.get(fld_names.GEN_VALIDATION_FILES, "false"))
    activate_transfer: bool = cast_to_bool(dc_settings.get(fld_names.ACTIVATE_TRANSFER, "false"))
    activate_hourly_workers: bool = cast_to_bool(dc_settings.get(fld_names.ACTIVATE_HOURLY, "true"))
    cost_polyvalents: float = float(dc_settings.get("cost_polyvalents", 0.001))

    hourly_workers_cost: float = float(dc_settings.get(fld_names.HOURLY_COST, 99.0))
    hourly_work_force: float = float(dc_settings.get(fld_names.HOURLY_FORCE, 7.0))

    def __str__(self):
        var_names = ["anticipate_backlog",
                     "fix_cpts_outbound",
                     "fix_slas_inbound",
                     "generate_validation_files",
                     "activate_transfer",
                     "activate_hourly_workers",
                     "hourly_workers_cost",
                     "hourly_work_force"]

        return "\nConfiguration:\n" + "\n".join(f"{v} {eval('Configuration.' + v)}" for v in var_names)
