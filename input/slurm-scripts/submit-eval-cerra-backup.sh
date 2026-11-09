#!/bin/bash
# Submit wrapper for the evaluation job: derives job name and log file names
# from the config file name (everything after "eval-cnn-cerra-").
#
# Template variables used in all names below:
#   @VAR@         the climate variable (t2m, rh, ...)
#   @REGION_PART@ expands to "_<region>" if REGION is set, otherwise to ""
#                  -> "cerra_dm_t2m_ger" vs "cerra_dm_t2m" (no dangling "_")
#   @REGION_DASH@ expands to "-<region>" if REGION is set, otherwise to ""
#                  -> config "eval-cnn-cerra-t2m-ger_..." vs "...-t2m_..."
# All placeholders are expanded after all overrides are read.
#
# Usage (config file and KEY=VALUE overrides in any order, config optional):
#   ./submit_eval.sh                                        # VAR=t2m, REGION=ger
#   ./submit_eval.sh VAR=rh                                 # same job for rh
#   ./submit_eval.sh REGION=eu                              # another region
#   ./submit_eval.sh REGION=                               # no region: cerra_dm_t2m
#   ./submit_eval.sh VAR=rh REGION=eu START_YEAR=2020 END_YEAR=2020
#   ./submit_eval.sh eval-cnn-cerra-t2m-ger_sort_en4_po1_batch.txt
#
# Accepted KEY=VALUE overrides:
#   VAR REGION CONFIG_FILE BASENAME SUFFIX TARGET_SUFFIX EXT START_YEAR END_YEAR
#   DATA_DIR MASK_DIR MASK_FILE STEADY_MASK_FILE XR_TEMPLATE
#
# For VAR=t2m, REGION=ger (default) this submits:
#   job name : crai-eval-t2m-ger_sort_en4_po1_batch   (derived from config name)
#   stdout   : .../eval_slurm_output/eval-cnn-t2m-ger_sort_en4_po1_batch.o<jobid>
#   stderr   : .../eval_slurm_output/eval-cnn-t2m-ger_sort_en4_po1_batch.e<jobid>

set -euo pipefail

# --- wrapper settings --------------------------------------------------------
PREFIX="eval-cnn-cerra-"
JOB_SCRIPT="slurm-eval-crai.sh"   # the sbatch file with the SBATCH resource headers
LOG_DIR="/work/uo1075/u244085/trees_in_the_city/ML_downscaling/Physics-Informed-CNN-Reconstructions/input/slurm-scripts/eval_slurm_output"
JOB_NAME_PREFIX="crai-eval"
LOG_PREFIX="eval-cnn"

# --- job settings: templates (@VAR@, @REGION_PART@, @REGION_DASH@) -----------
VAR="t2m"      # climate variable, e.g. t2m, rh, ...
REGION="ger"   # region, e.g. ger, eu; set to "" (REGION=) for names without region

MODEL_NAMES="best"    # checkpoint to evaluate: best, final, or an iteration number like 300000
CONFIG_TEMPLATE="eval-cnn-cerra-@VAR@@REGION_DASH@_sort_en4_po1_batch.txt"  # used if no config file is given
BASENAME="cerra_dm_@VAR@@REGION_PART@"
SUFFIX="_lowres_values_in_highres_grid"   # input files: <BASENAME>_<year><SUFFIX>.nc
TARGET_SUFFIX=""                           # target files: <BASENAME>_<year><TARGET_SUFFIX>.nc
EXT=".nc"
START_YEAR=2016
END_YEAR=2025
DATA_DIR="/work/uo1075/u244085/cerra_data/@VAR@/"   # dir holding the data .nc files
MASK_DIR="/work/uo1075/u244085/cerra_data/mask/"
MASK_FILE="cerra_dm_@VAR@@REGION_PART@_2025_mask.nc"     # per-timestep masks; must match data length per year
STEADY_MASK_FILE="cerra_dm_@VAR@@REGION_PART@_2025_steady_mask.nc"
XR_TEMPLATE=""   # empty -> job script builds it from DATA_DIR/BASENAME/START_YEAR/SUFFIX

# all settings passed to the job script
ALLOWED="VAR REGION CONFIG_FILE BASENAME SUFFIX TARGET_SUFFIX EXT START_YEAR END_YEAR DATA_DIR MASK_DIR MASK_FILE STEADY_MASK_FILE XR_TEMPLATE MODEL_NAMES"
# --- parse arguments: config file (positional) + KEY=VALUE overrides -----------
CONFIG_ARG=""
CONFIG_FILE="${CONFIG_FILE:-}"

for arg in "$@"; do
    if [[ "$arg" == *=* ]]; then
        key="${arg%%=*}"
        value="${arg#*=}"

        if [[ -z "$key" ]]; then
            echo "ERROR: '$arg' is not a valid KEY=VALUE override." >&2
            exit 1
        fi
        if ! grep -qw "$key" <<<"$ALLOWED"; then
            echo "ERROR: unknown setting '$key'. Allowed: $ALLOWED" >&2
            exit 1
        fi

        # set the variable inside this wrapper (also handles values with spaces)
        printf -v "$key" '%s' "$value"
        echo "  override: ${key}=${value}"
    else
        if [[ -n "$CONFIG_ARG" ]]; then
            echo "ERROR: config file specified more than once: '$arg'" >&2
            exit 1
        fi
        CONFIG_ARG="$arg"
    fi
done

# --- expand all placeholders (after all overrides are read) -------------------
# region parts: "_ger" / "-ger" if REGION is set, "" if REGION is empty
REGION_PART="${REGION:+_${REGION}}"

expand() {
    local s="$1"
    s="${s//@VAR@/$VAR}"
    s="${s//@REGION_PART@/$REGION_PART}"
    printf '%s' "$s"
}

BASENAME=$(expand "$BASENAME")
DATA_DIR=$(expand "$DATA_DIR")
MASK_FILE=$(expand "$MASK_FILE")
STEADY_MASK_FILE=$(expand "$STEADY_MASK_FILE")
XR_TEMPLATE=$(expand "$XR_TEMPLATE")
CONFIG_TEMPLATE=$(expand "$CONFIG_TEMPLATE")

# --- resolve config file: positional arg > CONFIG_FILE override > template ----
if [[ -n "$CONFIG_ARG" ]]; then
    CONFIG_FILE="$CONFIG_ARG"
else
    CONFIG_FILE="${CONFIG_FILE:-$CONFIG_TEMPLATE}"
fi

# strip directory part if a path was given
BASENAME_CFG=$(basename "$CONFIG_FILE")

# strip extension (.txt or anything else)
TAG="${BASENAME_CFG%.*}"

# strip the leading "eval-cnn-cerra-" prefix
TAG="${TAG#$PREFIX}"

# fallback if the prefix was not present or nothing is left
if [[ -z "$TAG" ]]; then
    TAG="default"
fi

CONFIG_FILE="$BASENAME_CFG"

# --- build the export list from ALL settings ----------------------------------
EXPORT_ARGS=""
for key in $ALLOWED; do
    EXPORT_ARGS+="${key}=${!key},"
done
EXPORT_ARGS="${EXPORT_ARGS%,}"

echo "Submitting evaluation job for config: $CONFIG_FILE"
echo "  job name : ${JOB_NAME_PREFIX}-${TAG}"
echo "  logs     : ${LOG_DIR}/${LOG_PREFIX}-${TAG}.o<jobid> / .e<jobid>"
echo "  variable : ${VAR}   region: '${REGION}'"
echo "  years    : ${START_YEAR}..${END_YEAR}"
echo "  data dir : ${DATA_DIR}"
echo "  basename : ${BASENAME}"

# --- submit -------------------------------------------------------------------
# Job name and log names cannot be set from $1 inside #SBATCH lines (those are
# read at submission time, before the shell runs), so they are passed here.
# --export=ALL forwards every setting above to the job script, where each
# settings line uses ${VAR:-default}. The config file is additionally passed
# as positional argument for direct sbatch runs of the job script.
sbatch \
    --job-name="${JOB_NAME_PREFIX}-${TAG}" \
    --output="${LOG_DIR}/${LOG_PREFIX}-${TAG}.o%j" \
    --error="${LOG_DIR}/${LOG_PREFIX}-${TAG}.e%j" \
    --export="ALL,${EXPORT_ARGS}" \
    "$JOB_SCRIPT" "$CONFIG_FILE"