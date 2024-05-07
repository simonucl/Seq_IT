INPUT_FILE=self-seq/data/alpaca/alpaca_llmam_70b.jsonl

python3 self-seq/gpt-query.py \
    --input_file $INPUT_FILE \
    --query meta-llama/Meta-Llama-3-70B-Instruct \
    --batch_size 4 \
    --use_instruct \
    --use_vllm \
    --no_refinement \
    --iteration 
