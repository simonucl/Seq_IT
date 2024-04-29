python3 self-seq/gpt-query.py \
    --input_file self-seq/data/flancot/flancot_filtered_15k.jsonl \
    --sample 200 \
    --query /mnt/data/models/c4ai-command-r-plus \
    --batch_size 4 \
    --load_4bit \
    --use_instruct

python3 self-seq/gpt-query.py \
    --input_file self-seq/data/alpaca/alpaca-cleaned.jsonl \
    --sample 200 \
    --query /mnt/data/models/c4ai-command-r-plus \
    --batch_size 4 \
    --load_4bit \
    --use_instruct