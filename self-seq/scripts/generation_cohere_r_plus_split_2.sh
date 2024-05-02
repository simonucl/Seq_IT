INPUT_FILE=self-seq/data/alpaca/alpaca-split_2.jsonl

python3 self-seq/gpt-query.py \
    --input_file $INPUT_FILE \
    --query /mnt/data/models/c4ai-command-r-plus-GPTQ \
    --batch_size 4 \
    --use_instruct \
    --ignore_cache \
    --use_vllm \
    --no_refinement