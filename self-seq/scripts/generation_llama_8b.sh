
# python3 self-seq/gpt-query.py \
#     --input_file self-seq/data/flancot/flancot_filtered_15k.jsonl \
#     --sample 50 \
#     --query gpt-3.5-turbo \
#     --batch_size 16 \
# 
# python3 self-seq/gpt-query.py \
#     --input_file self-seq/data/flancot/flancot_filtered_15k.jsonl \
#     --sample 50 \
#     --query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct \
#     --batch_size 1 \
#     --use_vllm

python3 self-seq/gpt-query.py \
    --input_file self-seq/data/flancot/flancot_filtered_15k.jsonl \
    --sample 20 \
    --query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-8B \
    --batch_size 16
