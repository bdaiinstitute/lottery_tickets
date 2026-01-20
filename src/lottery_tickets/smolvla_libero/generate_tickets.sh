#!/bin/bash

NUM_TICKETS=1
TASK=libero_spatial
EPSILON=None
BATCH_SIZE=1

for arg in "$@"; do
    case $arg in
        --num_tickets=*)
            NUM_TICKETS="${arg#*=}"
            ;;
        --task=*)
            TASK="${arg#*=}"
            ;;
        --output_dir=*)
            OUTPUT_PATH="${arg#*=}"
            ;;
        --num_episodes=*)
            NUM_EPISODES="${arg#*=}"
            ;;
        --batch_size=*)
            BATCH_SIZE="${arg#*=}"
            ;;
        --epsilon=*)
            EPSILON="${arg#*=}"
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# Required args check
: "${OUTPUT_PATH:?Missing --output_dir}"
: "${NUM_EPISODES:?Missing --num_episodes}"

echo "Generating $NUM_TICKETS tickets with epsilon=$EPSILON to $OUTPUT_PATH"

for i in $(seq 1 "$NUM_TICKETS"); do
    seed=$((1000+i))
    echo "Generating ticket $i of $NUM_TICKETS..."
    uv run python evaluate.py --policy.path="HuggingFaceVLA/smolvla_libero"  --env.type=libero --env.task="$TASK"  --eval.batch_size="$BATCH_SIZE"  --eval.n_episodes="$NUM_EPISODES" --output_dir="$OUTPUT_PATH" --eval_mode=NEW_TICKET  --seed=$seed --epsilon="$EPSILON"
done

echo "Successfully generated $NUM_TICKETS tickets!"
