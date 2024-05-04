INPUT_FILE=self-seq/data/alpaca_original/alpaca-split_2-ori.jsonl

python3 self-seq/gpt-query.py \
    --input_file $INPUT_FILE \
    --query /mnt/data/models/c4ai-command-r-plus-GPTQ \
    --batch_size 4 \
    --use_instruct \
    --regen_response \
    --use_vllm \
    --no_refinement