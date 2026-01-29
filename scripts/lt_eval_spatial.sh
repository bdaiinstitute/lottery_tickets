#!/bin/bash
#SBATCH --job-name=ticket_eval
#SBATCH -c 10            # minimum number of cores
#SBATCH -t 0-12:00:00   # time in d-hh:mm:ss
#SBATCH --mem=120G
#SBATCH -p public      # partition
#SBATCH -q public      # QOS
#SBATCH --gres=gpu:a100:1
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="%u@asu.edu"

# Activate your virtual environment
source activate lottery_tickets
cd $HOME/src/lottery_tickets/src/lottery_tickets/smolvla_libero/

# Use noise_path from argument if provided, otherwise use default
NOISE_PATH="${1:-/home/opatil3/src/lottery_tickets/src/lottery_tickets/smolvla_libero/golden_tickets/libero_spatial_tickets/ecf051d3ed474433a505d46dc007e9a6/initial_noise.pt}"

python evaluate.py \
    --policy.path="HuggingFaceVLA/smolvla_libero" \
    --env.type=libero \
    --env.task=libero_spatial \
    --eval.batch_size=10 \
    --eval.n_episodes=50 \
    --output_dir=outputs/libero_spatial_tickets/ticket_results \
    --eval_mode=LOAD_TICKET \
    --seed=1619 \
    --noise_path "$NOISE_PATH"