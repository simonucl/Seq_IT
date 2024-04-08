import pandas as pd
from datasets import load_dataset

import argparse
import json
import random
from ..token import HF_TOKEN

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--data', type=str, default='yahma/alpaca-cleaned')
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--sample', type=int, default=None)

    args = args.parse_args()
    dataset = load_dataset(args.data, split='train', token=HF_TOKEN)
    dataset_name = args.data.split('/')[-1]
    random.seed(args.seed)
    if 'idx' not in dataset.column_names:
        idx = list(range(len(dataset)))
        dataset = dataset.add_column('idx', idx)
    dataset = dataset.shuffle()
    if args.sample:
        dataset = dataset.select(range(args.sample))
        dataset.to_json(f'data/{dataset_name}_{args.sample}.jsonl', orient='records', lines=True)
    else:
        dataset.to_json(f'data/{dataset_name}.jsonl', orient='records', lines=True)
    
    # with open(input_file, 'r', encoding='utf-8') as file:
    #     data = []
    #     for line in file:
    #         data.append(json.loads(line))

    # df = pd.DataFrame(data)
    # df.to_csv(output_file, index=False)