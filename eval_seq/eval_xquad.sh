#!/bin/bash

for FILE in eval_results/self-seq-7B-1-3-new_fewshot/self-seq-7B-1-3-new_fewshot eval_results/self-seq-7B-baseline_fewshot/self-seq-7B-baseline_fewshot
do
    for LANG in en es ar el hi th de ru zh tr vi
    do
        python3 eval_seq/eval_xquad.py \
            --test_file ${FILE}_base_xquad_${LANG}.jsonl \
            --ref_file data/xquad/fewshot/xquad_en.jsonl
    done

    # for LANG in ar hi th tr vi
    # do
    #     python3 eval_seq/eval_xquad.py \
    #         --test_file ${FILE}_trans_xquad_${LANG}.jsonl \
    #         --ref_file ${FILE}_base_xquad_en.jsonl
    # done

done