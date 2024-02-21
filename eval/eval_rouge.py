import json
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
predict_es =[]
#load the jsonl file
with open('../multilingual-alpaca/commonsense_qa_pararepeatpara_llama7b-alpaca.jsonl', 'r') as file:
    for line in file:
        obj = json.loads(line)
        predict_es.append(obj)
#extract 'text' from each item of the jsonl file
preds = []
for i in range(len(predict_es)):
    preds.append(predict_es[i]['output'])

labels = []
for i in range(len(predict_es)):
    labels.append(predict_es[i]['input'])
#load json file
with open('../multilingual-alpaca/commonsense_qa_repeat_answer.json') as f:
    lines = json.load(f)
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
    score += sentence_bleu(labels[i], preds[i], smoothing_function=smoothie)
print(score/len(preds))

from rouge import Rouge
rouge = Rouge()
score = 0
for i in range(len(preds)):
    score += rouge.get_scores(preds[i], labels[i])[0]['rouge-l']['f']
print(score/len(preds))
from bert_score import score
P, R, F1 = score(preds, labels, lang="en", verbose=True)
print(F1.mean())
#print(score/len(preds))
