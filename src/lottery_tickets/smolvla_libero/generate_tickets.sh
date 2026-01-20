#!/bin/bash

set -e

NUM_TICKETS=$1
EPSILON=$2
TASK=$3
OUTPUT_PATH=$4
shift 3


echo "Generating $NUM_TICKETS tickets with epsilon=$EPSILON to $OUTPUT_PATH"

for i in $(seq 1 "$NUM_TICKETS"); do
    seed=$((1000+i))
    echo "Generating ticket $i of $NUM_TICKETS..."
    uv run python evaluate.py --policy.path="HuggingFaceVLA/smolvla_libero"  --env.type=libero --env.task="$TASK"  --eval.batch_size=5  --eval.n_episodes=25 --output_dir="$OUTPUT_PATH" --eval_mode=NEW_TICKET  --seed=$seed --epsilon="$EPSILON"
done

echo "Successfully generated $NUM_TICKETS tickets!"
