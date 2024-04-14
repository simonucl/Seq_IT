import json
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from argparse import ArgumentParser

def main(args):
    references = []
    #load the jsonl file
    with open(args.ref_file, 'r') as file:
        for line in file:
            obj = json.loads(line)
            references.append(obj['output'])
    #load json file
    preds = []
    with open(args.test_file, 'r') as file:
        for line in file:
            obj = json.loads(line)
            preds.append(obj['output'])
    #extract 'output' from each item of json

    #labels = []
    '''
    for i in range(len(lines)):
        labels.append(lines[i]['input'])
    '''
    #compute the BLEU score
    smoothie = SmoothingFunction().method4
    score = 0
    for i in range(len(preds)):
        score += sentence_bleu(references[i], preds[i], smoothing_function=smoothie)
    print(score/len(preds))

    from rouge import Rouge
    rouge = Rouge()
    score = 0
    for i in range(len(preds)):
        score += rouge.get_scores(preds[i], references[i])[0]['rouge-l']['f']
    print(score/len(preds))
    from bert_score import score
    P, R, F1 = score(preds, references, lang="en", verbose=True)
    print(F1.mean())
#print(score/len(preds))
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test_file', type=str, default=None)
    parser.add_argument('--ref_file', type=str, default=None)
    args = parser.parse_args()
    main(args)
