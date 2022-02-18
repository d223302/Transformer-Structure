#!/usr/bin/env python3
import numpy as np
from transformers import RobertaForMaskedLM, BertTokenizer
import os
import argparse
import torch
from torch import nn

def entropy(p):
  p /= p.sum()
  E = p * torch.log(1/p + 1e-20)
  return E.detach().cpu().numpy().sum() 

def MI(x, y):
  x = -torch.log(x + 1e-20) * x
  x = x.sum(-1)
  y = -torch.log(y + 1e-20) * y
  y = y.sum(-1)
  return x - y

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', help = "evaluation data directory")
parser.add_argument('--model_dir', help = "evaluated model's path")
parser.add_argument('--tokenizer_path', help = "Tokenzier path")
parser.add_argument('--output_dir', help = "Output result directory")
args = parser.parse_args()

tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path,
  unk_token = '<unk>',
  sep_token = '</s>',
  pad_token = '<pad>',
  cls_token = '<s>',
  mask_token = '<mask>',
  eos_token = '</s>',
  bos_token = '<s>'
)
model = RobertaForMaskedLM.from_pretrained(args.model_dir)
model.eval()
if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'
model.to(device)

softmax = nn.Softmax(dim = -1).to(device)

line_count = 0
result = []
all_entropy = []
max_entropy = []
with open(os.path.join(args.data_dir, 'eval.txt'), 'r') as f:
  while True:
    line = f.readline()
    if line == "":
      break
    line = line.strip()
    if line == "":
      continue
    if line_count > 1000:
      break
    if len(line.split()) > 120:
      line = " ".join(line.split()[:120])
    input_ids = tokenizer(line, return_tensors = 'pt')['input_ids']
    seq_len = input_ids.shape[1]
    if seq_len < 85:
      continue
    if seq_len > 140:
      input_ids = input_ids[:128]
    line_count += 1
    input_ids = input_ids.repeat(81, 1)
    #input_ids[:, seq_len // 2] = tokenizer.mask_token_id
    for i, j in enumerate(range(seq_len // 2 + 2 - 40, seq_len // 2 + 2 + 41)):
      input_ids[i, j] = tokenizer.mask_token_id
    with torch.no_grad():
      logits = model(input_ids.to(device))[0]
      gt = softmax(logits)
      input_ids[:, seq_len // 2 + 2] = tokenizer.mask_token_id
      logits = model(input_ids.to(device))[0]
      prob = softmax(logits)
      gt = gt[:, (seq_len // 2 + 2 - 40): (seq_len // 2 + 2 + 41)]
      prob = prob[:, (seq_len // 2 + 2 - 40): (seq_len // 2 + 2 + 41)]
      mi = MI(gt, prob)
      mi = mi.diag()
      all_entropy.append(entropy(softmax(mi)))
      mi[40] = -10000
      max_idx = mi.argmax()
      mi = torch.zeros_like(mi)
      mi[max_idx] = 1
      max_entropy.append(entropy(gt[max_idx, max_idx]))
      assert not np.isnan(max_entropy[-1])
      result.append(mi.detach().cpu().numpy())
result = np.stack(result)
#result /= result.max()
result = np.sum(result, 0)
result = [str(x) for x in result]
print("\t".join(result))
print(np.mean(all_entropy))
print(np.mean(max_entropy))
