INPUT_FILE=self-seq/data/alpaca_test/alpaca_test2.jsonl

python3 self-seq/gpt-query.py \
    --input_file $INPUT_FILE \
    --query /mnt/data/models/c4ai-command-r-plus-GPTQ \
    --batch_size 4 \
    --use_instruct \
    --use_vllm \
    --no_refinement
