#!/usr/bin/env bash
# ====================================================================
# submit.sh — top-level launcher
# ====================================================================
# The ONLY file you edit day-to-day: it holds all the variables
# (job name, file locations, years, parameters, resources) and submits
# job.sbatch via sbatch, which in turn calls analysis.py.
#
# Usage:
#   ./submit.sh                                        # standard config file, if it exists
#   ./submit.sh --config NAME                          # NAME is looked up in the configs dir
#   ./submit.sh --config <path/to/file>                # any path is used as-is
#   ./submit.sh --config NAME --years 2016 2020        # config + overrides
#   ./submit.sh --no-config                            # ignore the standard config file
#   ./submit.sh --var t2m --data-tag <tag> --years 2016 2020   # classic mode
#
# The directory with your config files is set in CONFIG_DIR below;
# a bare ./submit.sh automatically uses
# ${CONFIG_DIR}/${DEFAULT_CONFIG_NAME} if that file exists.
#
# What the config file provides (all other entries are ignored):
#   VAR        <- last entry of --data-types   (the evaluation variable)
#   DATA_TAG   <- --run-name
#   YEAR_MIN   <- --start-year
#   YEAR_MAX   <- --end-year
#   MASK_FILE  <- --steady-mask-file, with {var} (short name, e.g. "tp"),
#                 {reg} and {year} substituted
#
# Command-line flags override the config file; the DEFAULT_* values
# below are used only when neither provides a value.
# ====================================================================

set -euo pipefail

# --------------------------------------------------------------------
# 1. DATA: variable, names and locations of the files
# --------------------------------------------------------------------
# Mapping: directory/short name  ->  variable name inside the files.
declare -A VAR_FILE_MAP=(
    ["t2m"]="2t"
    ["orography"]="orog"
    ["land_sea_mask"]="lsm"
    ["tp"]="tp"
    ["relative_humidity"]="2r"
    ["surface_pressure"]="sp"
    ["wind_speed"]="10si"
    ["wind_direction"]="10wdir"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # folder of this script

# STANDARD DIRECTORY that holds your config files:
#   --config NAME  -> resolved to ${CONFIG_DIR}/NAME
#   ./submit.sh    -> automatically uses ${CONFIG_DIR}/${DEFAULT_CONFIG_NAME} if present
# Edit the CONFIG_DIR line to move the directory somewhere else
# (absolute paths work too, e.g. "/work/uo1075/u244085/.../configs").
CONFIG_DIR="${SCRIPT_DIR}/../input/levante"
DEFAULT_CONFIG_NAME="config.txt"
DEFAULT_CONFIG_FILE="${CONFIG_DIR}/${DEFAULT_CONFIG_NAME}"

# --------------------------------------------------------------------
# 1a. COMMAND-LINE ARGUMENTS (named flags, any order)
#     Empty string = "not set on the command line" -> config file wins
# --------------------------------------------------------------------
CONFIG_FILE=""
NO_CONFIG=""
VAR=""
DATA_TAG=""
YEAR_MIN=""
YEAR_MAX=""

# Defaults (used when neither the command line nor the config sets a value):
DEFAULT_VAR="t2m"
DEFAULT_DATA_TAG="cerra-germany-0.05x0.05-sorted"
DEFAULT_YEAR_MIN="2016"
DEFAULT_YEAR_MAX="2025"
DEFAULT_REGION="ger"

usage() {
    echo "Usage: $0 [--config FILE] [options]"
    echo ""
    echo "Options:"
    echo "  --config FILE     read VAR, DATA_TAG, years and mask from a config file"
    echo "                     (VAR = last entry of --data-types, DATA_TAG = --run-name)"
    echo "                     FILE may be a bare name (looked up in the configs"
    echo "                     directory) or any path"
    echo "  --no-config       ignore the standard config file"
    echo "  --var NAME        variable to evaluate, one of: ${!VAR_FILE_MAP[*]}"
    echo "  --data-tag TAG    dataset tag of the NN reconstruction"
    echo "  --year-min YEAR   first year to evaluate"
    echo "  --year-max YEAR   last year to evaluate"
    echo "  --years Y1 Y2     shorthand for --year-min Y1 --year-max Y2"
    echo "  -h, --help        show this help"
    echo ""
    echo "Defaults:  --var ${DEFAULT_VAR}  --data-tag ${DEFAULT_DATA_TAG}  --years ${DEFAULT_YEAR_MIN} ${DEFAULT_YEAR_MAX}"
    echo ""
    echo "Configs directory: ${CONFIG_DIR}"
    echo "Standard config file (used automatically if present):"
    echo "  ${DEFAULT_CONFIG_FILE}"
    echo ""
    echo "Examples:"
    echo "  $0                                # uses the standard config file if present"
    echo "  $0 --config myconfig.txt"
    echo "  $0 --config myconfig.txt --years 2016 2020"
    echo "  $0 --var t2m --data-tag cerra-germany-0.05x0.05-sorted --years 2016 2020"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config|-c)
            [[ $# -ge 2 ]] || { echo "ERROR: $1 needs a value" >&2; usage >&2; exit 1; }
            CONFIG_FILE="$2"; shift 2 ;;
        --no-config)
            NO_CONFIG="1"; shift 1 ;;
        --var|-v)
            [[ $# -ge 2 ]] || { echo "ERROR: $1 needs a value" >&2; usage >&2; exit 1; }
            VAR="$2"; shift 2 ;;
        --data-tag|-d)
            [[ $# -ge 2 ]] || { echo "ERROR: $1 needs a value" >&2; usage >&2; exit 1; }
            DATA_TAG="$2"; shift 2 ;;
        --year-min)
            [[ $# -ge 2 ]] || { echo "ERROR: $1 needs a value" >&2; usage >&2; exit 1; }
            YEAR_MIN="$2"; shift 2 ;;
        --year-max)
            [[ $# -ge 2 ]] || { echo "ERROR: $1 needs a value" >&2; usage >&2; exit 1; }
            YEAR_MAX="$2"; shift 2 ;;
        --years)
            [[ $# -ge 3 ]] || { echo "ERROR: --years needs two values: --years Y1 Y2" >&2; usage >&2; exit 1; }
            YEAR_MIN="$2"; YEAR_MAX="$3"; shift 3 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "ERROR: unknown option '$1'" >&2; usage >&2; exit 1 ;;
    esac
done

# --------------------------------------------------------------------
# 1b. VALUES FROM THE CONFIG FILE
#     (only for the settings that were NOT given on the command line)
# --------------------------------------------------------------------
# Print the value of flag "$1" from the config file; "" if absent.
# Handles "--flag value" on one line, "--flag=value", and the value on
# the following line.
cfg_get() {
    tr '\n\r=' '   ' < "${CONFIG_FILE}" | tr -s ' ' | awk -v key="$1" '
        {
            n = split($0, t, " ")
            for (i = 1; i < n; i++)
                if (t[i] == key) { print t[i+1]; exit }
        }'
}

# --config FILE: an existing path (absolute or relative) is used as-is;
# a bare name is looked up in the configs directory.
if [[ -n "${CONFIG_FILE}" && ! -f "${CONFIG_FILE}" && -f "${CONFIG_DIR}/${CONFIG_FILE}" ]]; then
    CONFIG_FILE="${CONFIG_DIR}/${CONFIG_FILE}"
fi

# Neither --config nor --no-config given? Fall back to the STANDARD
# config file, but only if it actually exists.
if [[ -z "${CONFIG_FILE}" && -z "${NO_CONFIG}" && -f "${DEFAULT_CONFIG_FILE}" ]]; then
    CONFIG_FILE="${DEFAULT_CONFIG_FILE}"
fi

if [[ -n "${CONFIG_FILE}" ]]; then
    if [[ ! -f "${CONFIG_FILE}" ]]; then
        echo "ERROR: config file not found: ${CONFIG_FILE}" >&2
        [[ "${CONFIG_FILE}" != */* ]] && echo "       (also looked in: ${CONFIG_DIR}/)" >&2
        exit 1
    fi

    # Evaluation variable = LAST entry of --data-types
    if [[ -z "${VAR}" ]]; then
        data_types="$(cfg_get --data-types)"
        if [[ -n "${data_types}" ]]; then
            var_file_cfg="${data_types##*,}"          # e.g. "tp"
            VAR=""
            for key in "${!VAR_FILE_MAP[@]}"; do
                if [[ "${VAR_FILE_MAP[${key}]}" == "${var_file_cfg}" ]]; then
                    VAR="${key}"
                    break
                fi
            done
            if [[ -z "${VAR}" ]]; then
                echo "ERROR: the last --data-types entry '${var_file_cfg}' is not a known variable." >&2
                echo "       Known variables:" >&2
                for key in "${!VAR_FILE_MAP[@]}"; do
                    echo "  ${key} -> ${VAR_FILE_MAP[${key}]}" >&2
                done
                exit 1
            fi
        fi
    fi

    if [[ -z "${DATA_TAG}" ]]; then
        v="$(cfg_get --run-name)"
        if [[ -n "${v}" ]]; then DATA_TAG="${v}"; fi
    fi
    if [[ -z "${YEAR_MIN}" ]]; then
        v="$(cfg_get --start-year)"
        if [[ -n "${v}" ]]; then YEAR_MIN="${v}"; fi
    fi
    if [[ -z "${YEAR_MAX}" ]]; then
        v="$(cfg_get --end-year)"
        if [[ -n "${v}" ]]; then YEAR_MAX="${v}"; fi
    fi
fi

# Final defaults (only used if neither CLI nor config provided a value)
VAR="${VAR:-${DEFAULT_VAR}}"
DATA_TAG="${DATA_TAG:-${DEFAULT_DATA_TAG}}"
YEAR_MIN="${YEAR_MIN:-${DEFAULT_YEAR_MIN}}"
YEAR_MAX="${YEAR_MAX:-${DEFAULT_YEAR_MAX}}"

VAR_FILE="${VAR_FILE_MAP[${VAR}]:-}"       # looked up from the map above

if [[ -z "${VAR_FILE}" ]]; then
    echo "ERROR: unknown VAR '${VAR}'. Known variables:" >&2
    for key in "${!VAR_FILE_MAP[@]}"; do
        echo "  ${key} -> ${VAR_FILE_MAP[${key}]}" >&2
    done
    exit 1
fi

PYTHON_SCRIPT="${SCRIPT_DIR}/4a_analysis_script.py"   # the Python code
SBATCH_SCRIPT="${SCRIPT_DIR}/4a_slurm_evaluation.sh"    # the Slurm script

PATH_ORIGINAL="/work/uo1075/u244085/cerra_data/${VAR}/"
PATH_NN="/work/uo1075/u244085/trees_in_the_city/ML_downscaling/Physics-Informed-CNN-Reconstructions/evaluations/${VAR}/${DATA_TAG}/data/"

# Results are written to  .../results/<VAR>/<run-name>/  (run-name = DATA_TAG),
# mirroring the structure of the NN evaluations directory.
OUTPUT_DIR="${SCRIPT_DIR}/results/${VAR}/${DATA_TAG}"   # figures, NetCDF, stats
LOG_DIR="${SCRIPT_DIR}/logs"                # log folder (auto-created)

# Job name is composed automatically, or hardcode it here:
JOB_NAME="eval_${VAR}_${DATA_TAG}_${YEAR_MIN}-${YEAR_MAX}"

# --------------------------------------------------------------------
# 2. PARAMETERS FOR THE PYTHON SCRIPT
# --------------------------------------------------------------------
RENAME_NN=""                               # e.g. "longitude:lon,latitude:lat" if NN coords differ
QUANTILES="0.05,0.25,0.5,0.75,0.95"        # error quantiles per grid cell
QUANTILE_STRIDE="1"                         # time stride for quantile maps
QQ_STRIDE="10"                              # time stride for QQ-plot samples
SCALE="1.0"                                 # scale applied to bias/RMSE/quantile MAPS
CHUNK_TIME="30"                            # dask chunk size along time
DESEASONALIZE="1"                           # 1 = remove day-of-year climatology from both
                                            #     fields before computing EV/correlation
                                            #     (needs >= 2 years; adds a second pass)

# --------------------------------------------------------------------
# 2b. OPTIONAL CELL MASK (NetCDF on the same or a similar lat/lon grid)
# --------------------------------------------------------------------
# Default mask (used without a config file). NOTE: this line uses ${VAR}
# (long name, e.g. t2m) and ${YEAR_MIN}; the config-file template below
# uses the SHORT name (e.g. 2t/tp) and its own year -> adjust if needed.
MASK_FILE="/work/uo1075/u244085/cerra_data/mask/cerra_dm_${VAR}_ger_${YEAR_MIN}_steady_mask.nc"
MASK_VAR=$VAR_FILE                               # empty = auto-detect (only variable in the file)
MASK_VALUE="0"                              # mask value that defines the selection
MASK_MODE="exclude"                            # "keep"    -> keep ONLY cells where mask == MASK_VALUE
                                            # "exclude" -> NaN the cells where mask == MASK_VALUE

# From the config file: use the --steady-mask-file template instead.
# {var} is replaced by the SHORT name (as in --data-types), {reg} by
# --region, and {year} (if present) by YEAR_MIN.
if [[ -n "${CONFIG_FILE}" ]]; then
    mask_tmpl="$(cfg_get --steady-mask-file)"
    if [[ -n "${mask_tmpl}" ]]; then
        region_cfg="$(cfg_get --region)"
        region_cfg="${region_cfg:-${DEFAULT_REGION}}"
        MASK_FILE="${mask_tmpl//\{var\}/${VAR}}"
        MASK_FILE="${MASK_FILE//\{reg\}/${region_cfg}}"
        MASK_FILE="${MASK_FILE//\{year\}/${YEAR_MIN}}"
    fi
fi

# --------------------------------------------------------------------
# 3. SLURM RESOURCE SETTINGS
# --------------------------------------------------------------------
PARTITION="compute"        # cluster partition
ACCOUNT="uo1075"                  # leave empty if your cluster needs none
TIME="02:00:00"             # walltime HH:MM:SS (5 years of quantiles take a while)
CPUS="8"                    # cpus per task (dask threads)
MEM="32G"                   # memory per node
MAIL_TYPE="START,END,FAIL"            # e.g. "END,FAIL"
MAIL_USER="mats.militzer@studium.uni-hamburg.de"                # your email, used only if

# --------------------------------------------------------------------
# 4. LOG FILES
# --------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${JOB_NAME}_%j.log"    # %j -> Slurm job ID

# --------------------------------------------------------------------
# 5. EXPORT VARIABLES FOR job.sbatch
# --------------------------------------------------------------------
export PYTHON_SCRIPT PATH_ORIGINAL PATH_NN VAR VAR_FILE YEAR_MIN YEAR_MAX
export OUTPUT_DIR RENAME_NN QUANTILES QUANTILE_STRIDE QQ_STRIDE SCALE CHUNK_TIME
export DESEASONALIZE
export MASK_FILE MASK_VAR MASK_VALUE MASK_MODE

# --------------------------------------------------------------------
# 6. BUILD THE SBATCH COMMAND
# --------------------------------------------------------------------
SBATCH_ARGS=(
    --job-name="${JOB_NAME}"
    --partition="${PARTITION}"
    --time="${TIME}"
    --cpus-per-task="${CPUS}"
    --mem="${MEM}"
    --output="${LOG_FILE}"        # stdout AND stderr go to the same log
    --export=ALL
)

# Optional settings: only add them when they are actually set
[[ -n "${ACCOUNT}"   ]] && SBATCH_ARGS+=(--account="${ACCOUNT}")
[[ -n "${MAIL_TYPE}" && "${MAIL_TYPE}" != "NONE" ]] && SBATCH_ARGS+=(--mail-type="${MAIL_TYPE}")
[[ -n "${MAIL_USER}" && "${MAIL_TYPE}" != "NONE" ]] && SBATCH_ARGS+=(--mail-user="${MAIL_USER}")

# --------------------------------------------------------------------
# 7. SUBMIT
# --------------------------------------------------------------------
echo "Submitting job '${JOB_NAME}' ..."
if [[ -n "${CONFIG_FILE}" ]]; then
    echo "  Config file  : ${CONFIG_FILE}"
fi
echo "  Reference dir : ${PATH_ORIGINAL}"
echo "  NN dir        : ${PATH_NN}"
echo "  Variable      : ${VAR} (internal name: ${VAR_FILE})"
echo "  Data tag      : ${DATA_TAG}"
echo "  Years         : ${YEAR_MIN}-${YEAR_MAX}"
echo "  Mask file     : ${MASK_FILE:-none}"
echo "  Output dir    : ${OUTPUT_DIR}"
echo "  Log file      : ${LOG_DIR}/${JOB_NAME}_<jobid>.log"
echo ""

sbatch "${SBATCH_ARGS[@]}" "${SBATCH_SCRIPT}" "${DATA_TAG}"