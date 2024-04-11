export CUDA_VISIBLE_DEVICES=0,1

MODEL_SIZE=7B
NUM_GPUS=1
BATCH_SIZE_PER_GPU=8
EVAL_BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=128
MODEL_NAME_OR_PATH=meta-llama/Llama-2-7b-hf
TRAIN_FILE=data/processed/tulu_v2_subset/tulu_v2_subset.jsonl

# get the model name from train file, and replace "_" with "-"
MODEL_NAME=$(basename $TRAIN_FILE .jsonl | tr _ -)
MODEL_NAME=Llama-2-7b-hf-${MODEL_NAME}
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    self-seq/finetune.py \
    --tracker_model_name $MODEL_NAME \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --use_flash_attn \
    --tokenizer_name $MODEL_NAME_OR_PATH \
    --use_slow_tokenizer \
    --train_file $TRAIN_FILE \
    --max_seq_length 2048 \
    --preprocessing_num_workers 24 \
    --checkpointing_steps epoch \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --use_lora \
    --lora_rank 8 \
    --lora_dropout 0.05 \
    --lora_alpha 8 \
    --do_eval \
    --eval_steps 100 \
    --eval_file data/processed/lima/lima_test_data.jsonl \
    --output_dir output/self-seq${MODEL_NAME} \
    --report_to wandb \
    --logging_steps 5

mkdir -p eval_results
