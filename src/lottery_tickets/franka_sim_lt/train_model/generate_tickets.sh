#!/usr/bin/env bash
# Defaults (optional)
N=1
NEW_NOISE=true

for arg in "$@"; do
    case $arg in
        --n=*)
            N="${arg#*=}"
            ;;
        --model_path=*)
            MODEL_PATH="${arg#*=}"
            ;;
        --output_dir=*)
            OUTPUT_BASE_DIR="${arg#*=}"
            ;;
        --num_episodes=*)
            NUM_EPISODES="${arg#*=}"
            ;;
        --new_noise=*)
            NEW_NOISE="${arg#*=}"
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# Required args check
: "${MODEL_PATH:?Missing --model_path}"
: "${OUTPUT_BASE_DIR:?Missing --output_dir}"
: "${NUM_EPISODES:?Missing --num_episodes}"

for ((i=1; i<=N; i++)); do
    python evaluate.py \
        evaluation.model_path="$MODEL_PATH" \
        +new_noise="$NEW_NOISE" \
        evaluation.num_episodes="$NUM_EPISODES" \
        hydra.run.dir="$OUTPUT_BASE_DIR/$i"
done
