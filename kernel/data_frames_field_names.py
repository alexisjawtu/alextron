"""
This file has the constants that store the names of the fields that the user
must write in the input of our linear optimizer.
"""

from typing import Dict, List

SHIFT_NAME: str = "shift_name"
SHIFT_TYPE: str = "shift_type"
SHIFT_TO: str = "shift_name_destination"
SHIFT_FROM: str = "shift_name_origin"

UNITARY_COST: str = "unitary_cost"
HIRING_COST: str = "hiring_cost"
DISMISSAL_COST: str = "dismissal_cost"
TRANSFER_COST: str = "cost"

INITIAL_WORKERS: str = "value"
MAX_WRKRS: str = "max_workers"
WRK_FORCE: str = "work_force"
WORK_DAY_NAME: str = "work_day_name"

# concerning absences.csv
UNJUSTIF_ABS_RATE: str = "unjustified_absent_rate"  # NOT PAID ABSENT RATE ?
JUSTIF_ABS_RATE: str = "justified_absent_rate"  # PAID ABSENT RATE
DAY: str = "day_name"

# concerning scheduling
VAL_START: str = "validity_start"
VAL_END: str = "validity_end"

HIRING_MODALITY: str = "contract_modality"
HIRING_TYPE: str = "contract_modality_type"

STAGE: str = "stage"
PROCESS: str = "process"
INBOUND_NAME: str = "inbound"
OUTBOUND_NAME: str = "outbound"

# Following the enumeration in the Process(Enum) class
STAGE_NAMES: Dict[int, List[str]] = {0: ["receiving", "checkin"], 1: ["picking", "packing"]}

PROC_ORIG: str = "process_origin"
PROC_DEST: str = "process_destination"

ID: str = "idx"
EPOCH_TIMESTAMP: str = "handling_ts"
EPOCH_ID: str = "handling_idx"
AUX_TS: str = "handling_ts_aux"

EH_FULL_RANGE: str = "eh_full_range"
START_MINS: str = "start_minute"
END_MINS: str = "end_minute"
MAX_EH_PER_DAY: str = "max_daily_extra_hours"

MIN_EH_BEFORE_SH: str = "start_prev"
MAX_EH_BEFORE_SH: str = "end_prev"

MIN_EH_AFTER_SH: str = "start_post"
MAX_EH_AFTER_SH: str = "end_post"

# Extra hours use 1 - q iff standard hours use q, where q := minutes/60.
START_RATIO: str = "start_ratio"  # the fraction: minutes/60
COMPLEMENTARY_START_RATIO: str = "complementary_start_ratio"  # the fraction: 1 - minutes/60
END_RATIO: str = "end_ratio"  # the fraction: minutes/60
COMPLEMENTARY_END_RATIO: str = "complementary_end_ratio"  # the fraction: 1 - minutes/60

MAX_SH_EPOCH: str = "max_epoch_of_shift"
MIN_SH_EPOCH: str = "min_epoch_of_shift"

SHIFT_ID: str = "shift_idx"

OUTPUT_DAY: str = "day"

# Fields concerning CONFIG_FILE
ANTICIPATE_BACKLOG: str = "anticipate_backlog"
FIX_CPTS_OUTBOUND: str = "fix_cpts_outbound"
FIX_SLAS_INBOUND: str = "fix_slas_inbound"
GEN_VALIDATION_FILES: str = "generate_validation_files"
ACTIVATE_TRANSFER: str = "activate_transfer"
ACTIVATE_HOURLY: str = "activate_hourly_workers"
HOURLY_COST: str = "hourly_workers_cost"
HOURLY_FORCE: str = "hourly_work_force"

# Modality types
PERM_MODALITY: str = "permanent"
TEMP_MODALITY: str = "temporal"
DAILY_MODALITY: str = "daily"
ALL_MODALITY: str = "ALL"

ACTIVATE_EH: str = "activate_extra_hours"
