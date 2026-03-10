DATA_ROOT="../../EXPLORE-Dataset"
ANNO="../../EXPLORE-Dataset/anno_abnormal.json"

LLM="Qwen/Qwen3-8B"          # path to llm scorer
BERT="sentence-transformers/all-MiniLM-L6-v2"  # path to sbert

DESCRIPTION_FILE="../infer/infer_results/Qwen3-VL-8B-Instruct/single-step/Qwen3-VL-8B-Instruct.json"
OUTPUT_DIR="./abn_scene_eval_res/Qwen3-VL-8B-Instruct"

DEVICE="cuda:0"

python eval_abn.py \
  --data_root "$DATA_ROOT" \
  --anno "$ANNO" \
  --llm "$LLM" \
  --bert "$BERT" \
  --soft_coverage \
  --description_file "$DESCRIPTION_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE"