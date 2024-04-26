
CHECKPOINT_PATH=simonycl/self-seq-Llama-2-7b-hf-new

MODEL_NAME=$(basename $CHECKPOINT_PATH)
# AlpacaEval
python3 -m eval.alpaca_farm.run_eval \
    --model_name_or_path $CHECKPOINT_PATH \
    --save_dir results/alpaca_farm/${MODEL_NAME} \
    --eval_batch_size 20 \
    --max_new_tokens 2048 \
    --stop_id_sequences \n\n
