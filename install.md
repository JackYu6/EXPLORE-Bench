```shell
conda create --name explore python=3.10 -y
conda activate explore

cd EXPLORE-Bench
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

python -m spacy download en_core_web_lg  
### or
wget https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl
pip install path/to/en_core_web_lg-3.8.0-py3-none-any.whl
```