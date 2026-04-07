# Model Inference Documentation

## Step 1: Data Preparation

**Download the [dataset](https://huggingface.co/datasets/pengfei2025/EXPLORE-Dataset)** to your local directory and set the `OUTPUT_PATH` environment variable to point to that location.

## Step 2: Model Configuration

**Configure the model settings** by specifying the `MODEL_NAME` and `MODEL_PATH`. The available models include:
   - `"qwen3-vl"`
   - `"qwen2.5-vl"`  
   - `"qwen2-vl"`  
   - `"ovis2.5"`  
   - `"minicpm-v4.5"`  
   - `"keye-vl1.5"`  
   - `"mimo_vl2508"`  
   - `"internvl3.5"`  
   - `"llava_onevision1.5"`  
   - `"step3-vl"`
   - `"glm4.6v-flash"`
   - `"egothinker"`
   - `"embodiedreasoner"`

## Step 3: Execute Inference

Run: `bash infer.sh`.

Here is an example of [infer.sh](infer.sh):
```bash
#!/bin/bash

gpu_list="0,1,2,3,4,5,6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

DATASET_PATH="../../EXPLORE-Dataset"
ANNO_FILE="anno.json"

MODEL_NAME="qwen3-vl"
ENABLE_THINKING=0 # 1 or 0, only available if supported by the model
MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_PATH="./infer_results/Qwen3-VL-8B-Instruct"

INFER_STRATEGY="single-step"
WINDOW_SIZE=0
SEGMENT_NUM=1
ROLLOUT="single-rollout"

SEED=42

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python infer.py \
    --dataset_path "$DATASET_PATH" \
    --anno_file "$ANNO_FILE" \
    --output_path "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME" \
    --enable_thinking $ENABLE_THINKING \
    --model_path "$MODEL_PATH" \
    --infer_strategy "$INFER_STRATEGY" \
    --window_size $WINDOW_SIZE \
    --segment_num $SEGMENT_NUM \
    --rollout "$ROLLOUT" \
    --num_chunks $CHUNKS \
    --chunk_idx $IDX \
    --seed $SEED &
done
wait
```

To run stepwise reasoning in the single-turn inference, set `INFER_STRATEGY="multi-step"` and `ROLLOUT="single-rollout"`, then specify the `SEGMENT_NUM` or `WINDOW_SIZE`.

To run stepwise reasoning in the multi-turn inference, set `INFER_STRATEGY="multi-step"` and `ROLLOUT="multi-rollout"`, then specify the `SEGMENT_NUM` or `WINDOW_SIZE`.

Example configuration can be found in [infer_multi_step_segs.sh](infer_multi_step_segs.sh) and [infer_multi_step_wins.sh](infer_multi_step_wins.sh).

## 🔧 Support New Models

To support your own models, you can define a new class in the `models` folder. Below is a template to guide you:

```python
class YourOwnModel
    def __init__(self, model_path):
        # load your model and processor here
    def generate_outputs(self, messages_list):
        # input: a list of messages
        # output the model answers in a list
```