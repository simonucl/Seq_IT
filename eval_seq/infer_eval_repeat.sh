#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
# export MODEL_PATH=$1

mkdir -p data/testset

# multilingual
MODEL_NAME=llama-7b
TASK=csqa
for MODEL_PATH in /mnt/nfs/public/hf/models/meta-llama/Llama-2-7b-hf
do
  MODEL_NAME=$(basename $MODEL_PATH)
  MODEL_NAME=${MODEL_NAME}_${PROMPT_TYPE}

  echo MODEL_NAME: ${MODEL_NAME}

  mkdir -p eval_results/${MODEL_NAME}

  TESTFILE=data/test/commonsense_qa_repeat.json
  python3 eval_seq/generate_batch.py \
    --base_model ${MODEL_PATH} \
    --length 128 \
    --test_file ${TESTFILE} \
    --batch_size 64 \
    --samples 100 \
    --save_file eval_results/${MODEL_NAME}/${MODEL_NAME}_csqa_repeat.json \
    --load_8bit False \
    --use_vllm True
done

# self-seq-7B-1-3-new self-seq-alpaca-cleaned_wizardlm_replaced