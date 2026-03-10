#!/bin/bash

DATA_ROOT="../../EXPLORE-Dataset"
LLM="Qwen/Qwen3-8B"          # path to llm scorer
BERT="sentence-transformers/all-MiniLM-L6-v2"  # path to sbert

DESCRIPTION_FILE="../infer/infer_results/Qwen3-VL-8B-Instruct/single-step/Qwen3-VL-8B-Instruct.json"
OUTPUT_DIR="./scene_eval_res/Qwen3-VL-8B-Instruct"

INFER_STRATEGY="single-step" # keep it consistent with inference
ROLLOUT="single-rollout"     # keep it consistent with inference
WINDOW_SIZE=0                # keep it consistent with inference
SEGMENT_NUM=1                # keep it consistent with inference 
EVAL_MODE="single-scene"
WHICH_SCENE="final"

NUM_PROCESSES=8
GPU_IDs="0,1,2,3,4,5,6,7"

declare -a ANNOS=(
  "../../EXPLORE-Dataset/anno_long_seq.json"
  "../../EXPLORE-Dataset/anno_medium_seq.json"
  "../../EXPLORE-Dataset/anno_short_seq.json"
  "../../EXPLORE-Dataset/anno.json"
)

declare -a DATASET_TYPES=(
  "long_seq"
  "medium_seq"
  "short_seq"
  "full"
)

for i in "${!ANNOS[@]}"; do
  ANNO="${ANNOS[$i]}"
  DATASET_TYPE="${DATASET_TYPES[$i]}"

  python eval.py \
    --data_root "$DATA_ROOT" \
    --anno "$ANNO" \
    --llm "$LLM" \
    --bert "$BERT" \
    --soft_coverage \
    --description_file "$DESCRIPTION_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --infer_strategy "$INFER_STRATEGY" \
    --eval_mode "$EVAL_MODE" \
    --window_size $WINDOW_SIZE \
    --segment_num $SEGMENT_NUM \
    --rollout "$ROLLOUT" \
    --which_scene "$WHICH_SCENE" \
    --dataset_type "$DATASET_TYPE" \
    --num_processes $NUM_PROCESSES \
    --gpu_ids "$GPU_IDs"
done