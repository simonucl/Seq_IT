
python3 self-seq/gpt-query.py \
    --input_file self-seq/data/flancot/flancot_filtered_15k.jsonl \
    --sample 200 \
    --query gpt-3.5-turbo 

python3 self-seq/gpt-query.py \
    --input_file self-seq/data/alpaca/alpaca-cleaned.jsonl \
    --sample 200 \
    --query gpt-3.5-turbo