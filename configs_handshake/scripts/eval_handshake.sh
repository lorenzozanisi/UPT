#!/bin/bash
# Evaluate the handshake models. One array task = one row of eval_combinations_handshake.txt.
#   sbatch --array=1-9  configs_handshake/scripts/eval_handshake.sh   # only what the plot needs
#   sbatch --array=1-15 configs_handshake/scripts/eval_handshake.sh   # also the two-END models
#
# Runs configs_handshake/code/eval_handshake.py (the April evaluation script) on each row.
# Each task writes plots/<trained_on>/<model_name>/<plots_name>/electron_temp_2d/<yaml_name>
# (plus per-sample PNGs). Set PLOTS_DIR to write somewhere other than <repo>/plots (which holds the April results).
#SBATCH --account=ukaea-ap001-gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=0:15:0
#SBATCH -p ukaea-amp
#SBATCH --mail-type=END,FAILED,BEGIN
#SBATCH --job-name=handshake_eval
#SBATCH --output=slurm-handshake-eval-%A_%a.out

UPT_ROOT=${UPT_ROOT:-/rds/project/rds-aSo1XX0UOlw/ir-zani1/UPT/UPT}
VENV=${VENV:-/home/ir-zani1/edge2d}
combinations_file=${COMBINATIONS:-$UPT_ROOT/configs_handshake/scripts/eval_combinations_handshake.txt}
CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-$UPT_ROOT/checkpoints/stage1}
PLOTS_DIR=${PLOTS_DIR:-$UPT_ROOT/plots}

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=55555

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl
source "$VENV/bin/activate"

# must run from src/ (static_config.yaml is read from the cwd) with src/ importable
cd "$UPT_ROOT/src"
export PYTHONPATH="$UPT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Read the (SLURM_ARRAY_TASK_ID)-th non-comment, non-empty line
line=$(grep -v '^[[:space:]]*#' "$combinations_file" | grep -v '^[[:space:]]*$' | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [ -z "$line" ]; then
    echo "No combination found for task ID ${SLURM_ARRAY_TASK_ID}"; exit 1
fi

read -r model_name trained_on variable yaml_name plots_name conditioning_vars_fname <<< "$line"
echo "Task ${SLURM_ARRAY_TASK_ID}: $line"

python "$UPT_ROOT/configs_handshake/code/eval_handshake.py" \
    --model_name "${model_name}" \
    --variable "${variable}" \
    --trained_on "${trained_on}" \
    --yaml_name "${yaml_name}" \
    --plots_name "${plots_name}" \
    --conditioning_vars_fname "${conditioning_vars_fname}" \
    --checkpoints_dir "${CHECKPOINTS_DIR}" \
    --plots_dir "${PLOTS_DIR}"
