
CHECKPOINT_PATH=$1

MODEL_NAME=$(basename $CHECKPOINT_PATH)

# check if sequential_instruction_tuning/SeqAlpacaEval/alpaca_eval_gpt4_baseline.seq.json exists
if [ -f "sequential_instruction_tuning/SeqAlpacaEval/alpaca_eval_gpt4_baseline.seq.json" ]; then
    echo "AlpacaEval file exists"
else
    git clone https://github.com/PinzhenChen/sequential_instruction_tuning.git
fi

python3 -m eval.alpaca_farm.run_seq_eval \
    --model_name_or_path $CHECKPOINT_PATH \
    --save_dir results/alpaca_farm/seqEval/${MODEL_NAME} \
    --eval_batch_size 20 \
    --max_new_tokens 2048 \
    --use_vllm \
    --prompt_path sequential_instruction_tuning/SeqAlpacaEval/seqEval.json \
    --use_chat_format \
    --chat_formatting_function tulu

python3 -m eval.alpaca_farm.reward \
        --input_file results/alpaca_farm/seqEval/${MODEL_NAME}/${MODEL_NAME}-seq-eval-greedy-long-output.json \
        --ref_file results/alpaca_farm/seqEval/gpt-3.5/gpt-3.5-turbo_reward.json