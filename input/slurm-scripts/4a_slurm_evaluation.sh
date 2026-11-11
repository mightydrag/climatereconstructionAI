#!/bin/bash
# ====================================================================
# job.sbatch — Slurm batch script
# ====================================================================
# Called by submit.sh. Input mode:
#   * ARGS_FILE set          -> config-file mode: analysis.py reads the
#                               variable, paths, years and mask from the
#                               arguments file itself.
#   * ARGS_FILE empty        -> manual mode: paths/years come from the
#                               exported variables below.
# All sbatch settings are passed by submit.sh as command-line options
# and override the #SBATCH directives (fallback defaults only).
# ====================================================================

#SBATCH --job-name=t2m_eval          # overridden by submit.sh
#SBATCH --partition=standard         # overridden by submit.sh
#SBATCH --time=06:00:00              # overridden by submit.sh
#SBATCH --cpus-per-task=8            # overridden by submit.sh
#SBATCH --mem=32G                     # overridden by submit.sh
#SBATCH --output=/dev/null           # overridden by submit.sh

source ~/.bashrc
#module load python3

mamba activate crai

set -euo pipefail

echo "=============================================================="
echo " Job ID        : ${SLURM_JOB_ID}"
echo " Job name      : ${SLURM_JOB_NAME}"
echo " Node          : ${SLURMD_NODENAME:-unknown}"
echo " CPUs          : ${SLURM_CPUS_PER_TASK:-1}"
echo " Args file     : ${ARGS_FILE:-<none, manual mode>}"
echo " Start time    : $(date)"
echo "=============================================================="

# ------------------------------------------------------------------
# Environment / modules (adjust to your cluster)
# NOTE: after `module purge`, check that your interpreter still exists
# (`python3` vs `python`) and that conda/mamba are reloaded if you need them.
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Safety check: everything submit.sh exports must be present.
# In config-file mode only PYTHON_SCRIPT and OUTPUT_DIR are required;
# in manual mode the data variables are required as well.
# ------------------------------------------------------------------
REQUIRED=(PYTHON_SCRIPT OUTPUT_DIR)
if [[ -z "${ARGS_FILE:-}" ]]; then
    REQUIRED+=(PATH_ORIGINAL PATH_NN VAR YEAR_MIN YEAR_MAX)
fi
for v in "${REQUIRED[@]}"; do
    if [[ -z "${!v:-}" ]]; then
        echo "ERROR: required variable '${v}' is not set." >&2
        echo "       Did you submit via submit.sh?" >&2
        exit 1
    fi
done

# optional metric switch (empty -> off)
DESEASONALIZE_FLAG=""
[[ "${DESEASONALIZE:-0}" == "1" ]] && DESEASONALIZE_FLAG="--deseasonalize"

# ------------------------------------------------------------------
# Input arguments: config file OR manual variables.
#
# --var-file / --mask-file / --mask-var are passed ONLY in manual mode:
# in config mode they would override the args file (--var-file forces
# the variable name and disables auto-detection; --mask-file overrides
# the --steady-masks entry of the args file).
#
# VAR_FILE_NN (variable name inside the NN output files, e.g.
# 'prediction' when the reference uses '2t') is valid in BOTH modes;
# empty -> auto-detected by analysis.py.
# ------------------------------------------------------------------
INPUT_ARGS=()
if [[ -n "${ARGS_FILE:-}" ]]; then
    INPUT_ARGS+=(--args-file "${ARGS_FILE}")
else
    INPUT_ARGS+=(--path-original "${PATH_ORIGINAL}")
    INPUT_ARGS+=(--path-nn "${PATH_NN}")
    INPUT_ARGS+=(--var "${VAR}")
    INPUT_ARGS+=(--year-min "${YEAR_MIN}")
    INPUT_ARGS+=(--year-max "${YEAR_MAX}")
    [[ -n "${VAR_FILE:-}"  ]] && INPUT_ARGS+=(--var-file "${VAR_FILE}")
    [[ -n "${MASK_FILE:-}" ]] && INPUT_ARGS+=(--mask-file "${MASK_FILE}")
    [[ -n "${MASK_VAR:-}"  ]] && INPUT_ARGS+=(--mask-var "${MASK_VAR}")
fi

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# ------------------------------------------------------------------
# Run the Python evaluation script
# MASK_VALUE / MASK_MODE apply in BOTH modes; the mask FILE is the
# --steady-masks entry of the args file (config mode) or MASK_FILE
# (manual mode only, see INPUT_ARGS above).
# File-name patterns default to the current naming:
#   reference : cerra_dm_t2m_ger_1986.nc             -> cerra_dm_{var}_ger_{year}.nc
#   NN output : cerra_dm_t2m_ger_2018_crai_output.nc  -> cerra_dm_{var}_ger_{year}_crai_output.nc
# (--file-pattern-ref is only used in manual mode; the reference files
#  in config mode come from the --test-names JSON of the args file.)
# ------------------------------------------------------------------
python3 "${PYTHON_SCRIPT}" \
    "${INPUT_ARGS[@]}" \
    --output-dir      "${OUTPUT_DIR}" \
    --var-file-nn     "${VAR_FILE_NN:-}" \
    --file-pattern-nn  "cerra_dm_{var}_ger_{year}_crai_output.nc" \
    --file-pattern-ref "cerra_dm_{var}_ger_{year}.nc" \
    --rename-nn       "${RENAME_NN:-}" \
    --quantiles       "${QUANTILES:-0.05,0.25,0.5,0.75,0.95}" \
    --quantile-stride "${QUANTILE_STRIDE:-1}" \
    --qq-stride       "${QQ_STRIDE:-10}" \
    --scale           "${SCALE:-1.0}" \
    --chunk-time      "${CHUNK_TIME:-30}" \
    --mask-value      "${MASK_VALUE:-1}" \
    --mask-mode       "${MASK_MODE:-keep}" \
    --point-radius-km "${POINT_RADIUS_KM:-0}" \
    --roc-quantiles   "${ROC_QUANTILES:-0.5,0.9,0.95,0.99}" \
    ${DESEASONALIZE_FLAG}

echo "=============================================================="
echo " Finished at : $(date)"
echo " Results in  : ${OUTPUT_DIR}"
echo "=============================================================="