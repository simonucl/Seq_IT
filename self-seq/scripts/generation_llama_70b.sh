python3 self-seq/gpt-query.py \
    --input_file self-seq/data/flancot/flancot_filtered_15k.jsonl \
    --sample 200 \
    --query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct \
    --batch_size 2 \
    --use_instruct

python3 self-seq/gpt-query.py \
    --input_file self-seq/data/alpaca/alpaca-cleaned.jsonl \
    --sample 200 \
    --query /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-70B-Instruct \
    --batch_size 2 \
    --use_instruct