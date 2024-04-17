#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
# export MODEL_PATH=$1

mkdir -p data/testset

# multilingual
MODEL_NAME=llama-7b
TASK=xquad
TRAIN_TYPE=base
for MODEL_PATH in llama-7b
do
  MODEL_NAME=$(basename $MODEL_PATH)
  MODEL_NAME=${MODEL_NAME}_fewshot

  echo MODEL_NAME: ${MODEL_NAME}

  mkdir -p eval_results/${MODEL_NAME}

  for TESTLANG in en es ar el hi th de ru zh tr vi
  do
    TESTFILE=data/xquad/fewshot/xquad_${TESTLANG}.jsonl
    python3 eval_seq/generate_batch.py \
      --base_model ${MODEL_PATH} \
      --length 1024 \
      --test_file ${TESTFILE} \
      --batch_size 64 \
      --save_file eval_results/${MODEL_NAME}/${MODEL_NAME}_${TRAIN_TYPE}_${TASK}_${TESTLANG}.jsonl \
      --load_8bit False \
      --use_vllm True
  done
done
