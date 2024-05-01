INPUT_FILE=self-seq/data/alpaca_test/alpaca_test1.jsonl

python3 self-seq/gpt-query.py \
    --input_file $INPUT_FILE \
    --query /mnt/data/models/c4ai-command-r-plus \
    --batch_size 4 \
    --load_4bit \
    --use_instruct \
    --no_refinement