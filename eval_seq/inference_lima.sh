#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export LORA_PATH=$1
export FINAL_SAVE=$2
# multilingual

MODEL=llama-7b
TESTFILE=self-seq/data/lima_500.jsonl
python3 eval/generate_batch.py \
  --base_model ${LORA_PATH} \
  --length 512 \
  --test_file ${TESTFILE} \
  --save_file ${FINAL_SAVE} \
  --batch_size 32 \
  --samples 100 \
