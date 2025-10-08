# Benchmark Dataset Evaluations

## Background

This directory contains information relevant for reproducing or evaluating system designs or prompts (comparing advertising engine with the control LLM).

## Setup

Link to LLM benchmark evaluation data: [https://drive.google.com/drive/folders/1a4xbkKwJ4UnzfFE6RuQVWBAkZezj7edQ?usp=drive_link](https://drive.google.com/drive/folders/1a4xbkKwJ4UnzfFE6RuQVWBAkZezj7edQ?usp=drive_link)

Download `results.tar.gz` and unzip it into the `eval/` directory. The directory structure should be `eval/outputs/`.

For the raw benchmark datasets, you can download all but the wildchat1m dataset in the link above. Download `benchmarks.tar.gz` and unzip it into any directory. Make sure that in `Config.py` that:
```python
DATA_DIR = Path('/your/path/to/llm_evals')
```
You can download the wildchat1m from [https://huggingface.co/datasets/allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M), but it is not part of the core evaluations that we run (only evaluating ad prevalence).

## Running Evals

```bash
cd eval/
python eval_normal/drop.py
python eval_normal/gpqa.py
python eval_normal/humaneval.py
python eval_normal/mgsm.py
python eval_normal/mtbenchmark.py
python eval_normal/testmath.py
python eval_normal/synthpai.py
```

These scripts will run the benchmark evaluations with their respective datasets. Modifying the `args` dictionary will change which model or ad engine configurations are evaluated.

## Results Analysis

```bash
cd eval/
python analyze_outputs.py
python eval_normal/synthpai_analyze.py
```

The performance results will be printed out, and the plots will appear in `eval/plots/`.