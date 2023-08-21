# NPCAI Llama Benchmark

This directory contains a benchmark script for Llama models. Please refer to this document for how to install a Llama model and run the benchmark script against it.

## Step 1: Install the repo and the model

See instructions for downloading the model of your choice [here](https://github.com/NPCAI-Studio/llama/blob/main/README.md).

## Step 2: Install dependencies

Run the follow command:

```
pip install -e .
```

## Step 3: Run the benchmark

Run the following command from the root directory of this repository:

```
torchrun --nproc_per_node 1 benchmarks/npcai_benchmark.py \
  --ckpt_dir llama-2-7b-chat/ \
  --tokenizer_path tokenizer.model
```
