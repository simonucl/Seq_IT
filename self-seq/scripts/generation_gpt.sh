
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
    --sample 150 \
    --query gpt-3.5-turbo 

python3 self-seq/gpt-query.py \
    --input_file self-seq/data/alpaca/alpaca-cleaned.jsonl \
    --sample 200 \
    --query gpt-3.5-turbo