import json
import numpy as np
from argparse import ArgumentParser
from tqdm import trange
import os
from rouge import Rouge

def main(args):
    references = []
    #load the jsonl file
    with open(args.ref_file, 'r', encoding='utf-8') as file:
        for line in file:
            obj = json.loads(line)
            references.append(obj)
    #load json file
    preds = []
    with open(args.test_file, 'r', encoding='utf-8') as file:
        for line in file:
            obj = json.loads(line)
            preds.append(obj)
    #extract 'output' from each item of json
    assert len(preds) == len(references), f'{len(preds)} != {len(references)}'
    rouge = Rouge()

    words = ['result:', 'answer:', 'answer is', 'result is']
    metric = {}
    r = 0
    ac_en = 0
    ac_tar = 0
    rouge_scores = []
    output_file = os.path.join(args.test_file.rsplit('/', 1)[0], 'metric.json')
    target_lang = args.test_file.replace('.jsonl', '').rsplit('_', 1)[1]

    for i in trange(len(preds), desc=f'Evaluating {target_lang}'):
        pred = preds[i]
        pred_output = pred['output'].lower()
        answer = references[i]
        target_answer = answer['target'].lower()
        target_lan_answer = pred['target'].lower()

        word = ''
        for word in words:
            position = pred_output.find(word)
            if position != -1:
                r+=1
                break
        pred = pred_output[position+len(word):].strip()
        if (target_answer in pred) or (pred in target_answer):
            ac_en += 1
        if (target_lan_answer in pred) or (pred in target_lan_answer):
            ac_tar += 1

        if (len(preds[i]['output']) == 0) or (len(references[i]['input']) == 0):
            rouge_scores.append(0)
        else:
            rouge_scores.append(
                rouge.get_scores(preds[i]['output'], references[i]['input'])[0]['rouge-l']['f']
            )
    
    metric['r'] = r
    metric['ac_en'] = ac_en / len(preds)
    metric['ac_tar'] = ac_tar / len(preds)
    metric['rouge'] = np.mean(rouge_scores)

    output_file = os.path.join(args.test_file.rsplit('/', 1)[0], 'metric.json')
    target_lang = args.test_file.replace('.jsonl', '').rsplit('_', 1)[1]
    data = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as file:
            data = json.load(file)
    data[target_lang] = metric
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test_file', type=str, default=None) # target language file
    parser.add_argument('--ref_file', type=str, default=None) # ref (usually en) language file
    args = parser.parse_args()
    main(args)
