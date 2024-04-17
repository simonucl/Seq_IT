#!/bin/bash

for FILE in eval_results/self-seq-7B-1-3-new/testset/llama-7b eval_results/self-seq-7B-baseline/testset/llama-7b eval_results/self-seq-alpaca-cleaned_wizardlm_replaced/testset/self-seq-alpaca-cleaned_wizardlm_replaced
do
    # for LANG in en de es ru de zh
    for LANG in zh
    do
        python3 eval_seq/eval_xquad.py \
            --test_file ${FILE}_base_xquad_${LANG}.jsonl \
            --ref_file ${FILE}_base_xquad_en.jsonl
    done

    # for LANG in ar hi th tr vi
    # do
    #     python3 eval_seq/eval_xquad.py \
    #         --test_file ${FILE}_trans_xquad_${LANG}.jsonl \
    #         --ref_file ${FILE}_base_xquad_en.jsonl
    # done

done