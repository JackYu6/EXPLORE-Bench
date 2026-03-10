# EXPLORE-Bench: Egocentric Scene Prediction with Long-Horizon Reasoning

<div align=center>

[![arXiv preprint](https://img.shields.io/badge/arxiv-2506.05287-ECA8A7?logo=arxiv)]() 
[![Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/JackYu6/EXPLORE-Dataset)
[![Project Page](https://img.shields.io/badge/🌐%20Project-Page-9DC3E6)](https://jackyu6.github.io/EXPLORE-Page/)

</div>

## 🔍 Overview
<p align="center">
    <img src="assets/teaser.jpg" width="100%" style="margin-bottom: 0.2;"/>
<p>

<strong>EXPLORE-Bench</strong> evaluates MLLMs on a new task: egocentric scene prediction with long-horizon reasoning. We annotate the final scene at the object, attribute, and relation levels to enable fine-grained scene-level evaluation. Note that the prompt is abbreviated for brevity in this figure.

## 🌟 Run Your Own Evaluation
### 🤗 Dataset
Our dataset is hosted on [HuggingFace](https://huggingface.co/datasets/JackYu6/EXPLORE-Dataset). 

### 🛠️ Installation
Set up your environment:

```shell
conda create --name explore python=3.10 -y
conda activate explore

git clone git@github.com:JackYu6/EXPLORE-Bench.git
cd EXPLORE-Bench

pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python -m spacy download en_core_web_lg
```

### 📚 parser and LLM evaluator
Download the weights of [Sentence-Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and [Qwen3](https://huggingface.co/Qwen/Qwen3-8B) from huggingface.

### ⚙️ Inference
Our codebase supports a variety of models for inference. Adjust the model settings in `infer/infer.sh`, then run the script to begin your inference.

To run your own models, you can add a class in the `infer/models` directory.

Please refer to our [inference documentation](infer/infer.md) for detailed guidance.

### 📈 Evaluation
Please refer to our [evaluation documentation](eval/eval.md).

## 📑 Citation
If you find our work helpful, please consider starring our repository and citing:
```

```

## ✨️ Acknowledgement
We sincerely thank the open-sourcing of these works where our code is based on:

[CompreCap](https://github.com/LuFan31/CompreCap) and [EOC-Bench](https://github.com/alibaba-damo-academy/EOCBench).