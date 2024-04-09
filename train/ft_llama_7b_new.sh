#!/bin/bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export CUDA_VISIBLE_DEVICES=$1

BASE_MODEL=meta-llama/Llama-2-7b-hf #baffo32/decapoda-research-llama-7B-hf #decapoda-research/llama-7b-hf
OUTPUT=alpaca-llama-2-7b-hf-new
TRAIN_TYPE=exp
DATA_PATH=../data/alpaca-cleaned_replaced.jsonl
mkdir models

OUTPUT_DIR=models/${TRAIN_TYPE}-${OUTPUT}
WORLD_SIZE=2 torchrun --nproc_per_node=2 --master_port 12343  train/finetune.py \
    --base_model ${BASE_MODEL} \
    --data_path ${DATA_PATH} \
    --output_dir ../${OUTPUT_DIR} \
    --batch_size 64 \
    --micro_batch_size 4 \
    --num_epochs 5 \
    --learning_rate 3e-4 \
    --cutoff_len 2048 \
    --val_set_size 500 \
    --lora_r 8 \
    --lora_alpha 8 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,k_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length \
    --use_wandb True \
    --wandb_run_name ${TRAIN_TYPE}-${OUTPUT}

