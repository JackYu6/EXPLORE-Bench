#!/bin/bash

gpu_list="0,1,2,3,4,5,6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

DATASET_PATH="../../EXPLORE-Dataset"
ANNO_FILE="anno.json"

MODEL_NAME="qwen3-vl"
MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_PATH="./infer_results/Qwen3-VL-8B-Instruct"

INFER_STRATEGY="multi-step"   
ROLLOUT="multi-rollout"       # "single-rollout" or "multi-rollout"
SEED=42

MIN_WINDOW=10
MAX_WINDOW=55

for WINDOW_SIZE in $(seq $MAX_WINDOW -5 $MIN_WINDOW); do
  echo "===== WINDOW_SIZE=${WINDOW_SIZE} ====="

  for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python infer.py \
      --dataset_path "$DATASET_PATH" \
      --anno_file "$ANNO_FILE" \
      --output_path "$OUTPUT_PATH" \
      --model_name "$MODEL_NAME" \
      --model_path "$MODEL_PATH" \
      --infer_strategy "$INFER_STRATEGY" \
      --window_size $WINDOW_SIZE \
      --rollout "$ROLLOUT" \
      --num_chunks $CHUNKS \
      --chunk_idx $IDX \
      --seed $SEED &
  done
  wait
done