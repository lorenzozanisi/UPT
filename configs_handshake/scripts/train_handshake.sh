#!/bin/bash
# Train ONE handshake model. Usage (from anywhere):
#   sbatch configs_handshake/scripts/train_handshake.sh configs_handshake/train/trained_on_CD_7lfha3pj.yaml
#
# Same launch arguments as the original April 2026 runs; the config is passed as an argument.
#SBATCH --account=ukaea-ap001-gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:0
#SBATCH -p ukaea-amp
#SBATCH --mail-type=END,FAILED,BEGIN
#SBATCH --exclude=gpu-q-60
#SBATCH --job-name=handshake_train

UPT_ROOT=${UPT_ROOT:-/rds/project/rds-aSo1XX0UOlw/ir-zani1/UPT/UPT}   # repo root (contains src/)
VENV=${VENV:-/home/ir-zani1/edge2d}                                   # python env used for the original runs

config=$(readlink -f "$1")
if [ -z "$1" ] || [ ! -f "$config" ]; then
    echo "usage: sbatch $0 <path/to/config.yaml>"; exit 1
fi

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=55555

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl
source "$VENV/bin/activate"

# main_train.py must run from src/: static_config.yaml, wandb_configs/ and zztemplates/ are resolved relative to it
cd "$UPT_ROOT/src"

options="--hp=$config \
        --accelerator=gpu \
        --devices=0 \
        --name=mastu_deuterium_only \
        --num_workers=32 \
        --wandb_mode=online \
        --master_port=$MASTER_PORT"

echo "python main_train.py $options"
python "$UPT_ROOT/src/main_train.py" $options
