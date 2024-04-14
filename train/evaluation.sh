# export CUDA_VISIBLE_DEVICES=0,1
HF_TOKEN=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

MODEL_SIZE=7B
NUM_GPUS=2
BATCH_SIZE_PER_GPU=2
EVAL_BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=64
MODEL_NAME_OR_PATH=meta-llama/Llama-2-7b-hf

# for MODEL in simonycl/sparseIT-Llama-2-7b-hf-multi-task simonycl/data_selection_Llama-2-7b-hf-multi_task-mask-mlp-by-dataset
# for MODEL in simonycl/sparseIT-Llama-2-7b-hf-multi-task
# for MODEL in simonycl/sparseIT_Llama-2-7b-hf-stanford-alpaca
# for MODEL in /mnt/data/sparseIT/output/sparseIT_Llama-2-7b-hf-stanford-alpaca-mask-by-cluster
# for MODEL in /mnt/data/sparseIT/output/sparseIT_Llama-2-7b-hf-multi-task-data-no-mlp
for MODEL in output/sparseIT_Llama-2-7b-hf-tulu-v2-subset-mask-by-cluster-row-0.1
do

    # split by '/' and select the last two elements and join them with '-'
    # MODEL_NAME=$(echo $MODEL | tr '/' '-' | cut -d '-' -f 2-)

    MODEL_NAME=$(basename "$MODEL")

    mkdir -p eval_results/
    
    # Run evaluation on ARC, GSM8K, HellaSwag, TruthfulQA, and TrivialQA
    bash lm-evaluation-harness/eval_model.sh $MODEL sparseIT_$MODEL_NAME > eval_results/sparseIT_$MODEL_NAME-1.log

    # Evaluation script for MMLU, TydiQA and CodeX-HumanEval
    bash scripts/eval/mmlu.sh $MODEL sparseIT_$MODEL_NAME > eval_results/sparseIT_$MODEL_NAME-mmlu.log
done

# nohup bash scripts/evaluation_lora.sh > ./logs/evaluation_lora_Llama-2-7b-hf-sharegpt-KMenasRandomDeita-64-005-lora-epoch_4.log 2>&1 &

# nohup bash scripts/evaluation_lora.sh > ./logs/evaluation_lora_Llama-2-7b-hf-sharegpt-KCenterGreedyDeita-005-lora-epoch_4.log 2>&1 &
