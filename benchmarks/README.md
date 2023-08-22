# NPCAI Llama Benchmark

This directory contains a benchmark script for Llama models. Please refer to this document for how to install a Llama model and run the benchmark script against it.

## Step 1: Install Python

Run the following commands:

```
sudo apt-get install python3.10
sudo apt install python3-pip
```

## Step 2: Install git

Run the follow command:

```
sudo apt-get update
sudo apt-get install git
```

## Step 3: Clone the repo and download the model

See instructions for downloading the model of your choice [here](https://github.com/NPCAI-Studio/llama/blob/main/README.md).

## Step 4: Install dependencies

Run the follow command:

```
pip install -e .
```

## Step 5: Run the benchmark

Run the following command from the root directory of this repository:

```
torchrun --nproc_per_node 1 benchmarks/npcai_benchmark.py \
  --ckpt_dir <model_dir>/ \
  --tokenizer_path tokenizer.model
```
