# export CUDA_VISIBLE_DEVICES=0,1
HF_TOKEN=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

MODEL_SIZE=7B
NUM_GPUS=2
BATCH_SIZE_PER_GPU=2
EVAL_BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=64
MODEL_NAME_OR_PATH=meta-llama/Llama-2-7b-hf

for MODEL in output/sparseIT_Llama-2-7b-hf-tulu-v2-subset-mask-by-cluster-row-0.1 #output/sparseIT_Llama-2-7b-hf-tulu-v2-mask-by-cluster-64-clusters-0-1-topk-closest
do

    # split by '/' and select the last two elements and join them with '-'
    # MODEL_NAME=$(echo $MODEL | tr '/' '-' | cut -d '-' -f 2-)

    MODEL_NAME=$(basename "$MODEL")

    # python3 finetune/merge_lora.py \
    # --base_model_name_or_path $MODEL_NAME_OR_PATH \
    # --lora_model_name_or_path $MODEL \
    # --output_dir output/sparseIT_${MODEL_NAME}_lora/ \
    # --tokenizer_name_or_path output/sparseIT_${MODEL_NAME}_lora/ \
    # --save_tokenizer

    # if not exist results/$MODEL_NAME, create it
    # mkdir -p results/$MODEL_NAME
    mkdir -p eval_results/
    
    # Run evaluation on ARC, GSM8K, HellaSwag, TruthfulQA, and TrivialQA
    bash lm-evaluation-harness/eval_model.sh $MODEL sparseIT_$MODEL_NAME > eval_results/sparseIT_$MODEL_NAME-1.log

    # Evaluation script for MMLU, TydiQA and CodeX-HumanEval
    # bash scripts/eval/mmlu.sh $MODEL sparseIT_$MODEL_NAME > eval_results/sparseIT_$MODEL_NAME-mmlu.log
done

# nohup bash scripts/evaluation_lora.sh > ./logs/evaluation_lora_Llama-2-7b-hf-sharegpt-KMenasRandomDeita-64-005-lora-epoch_4.log 2>&1 &

# nohup bash scripts/evaluation_lora.sh > ./logs/evaluation_lora_Llama-2-7b-hf-sharegpt-KCenterGreedyDeita-005-lora-epoch_4.log 2>&1 &
