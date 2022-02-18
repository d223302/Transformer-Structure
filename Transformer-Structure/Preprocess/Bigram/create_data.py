#!/usr/bin/env python3
import numpy as np
from tqdm import tqdm
freq = np.load("../En/freq.npy")
freq[:5] = 0
vocab = np.arange(freq.shape[0])
token_to_order = np.argsort(-freq) # order[i] = vocab of the ith most frequent token
order_to_token = np.zeros_like(token_to_order)
for i in range(order_to_token.shape[0]):
  order_to_token[token_to_order[i]] = i

unigram = np.load('../Bigram_En/unigram.npy')
bigram = np.load('../Bigram_En/bigram.npy')

random_index = np.arange(unigram.shape[0])
np.random.shuffle(random_index)
unigram = unigram[random_index]
P = np.eye(unigram.shape[0])[random_index]
bigram = np.matmul(np.matmul(P.T, bigram), P)

np.save('unigram.npy', unigram)
np.save('bigram.npy', bigram)

with open('data/train.txt', 'w') as f:
  for i in tqdm(range(930000)):
    sent = []
    sent.append(np.random.choice(a = np.arange(unigram.shape[0]), p = unigram))
    for pos in range(2, np.random.randint(low = 100, high = 120)): 
      row_sum = bigram[sent[-1]].sum()
      if row_sum > 0:
        sent.append(np.random.choice(a = np.arange(unigram.shape[0]), p = bigram[sent[-1]]))
      else:
        sent.append(np.random.choice(a = np.arange(unigram.shape[0]), p = unigram))
    sent = [str(order_to_token[x]) if x != 3000 else '3' for x in sent]
    f.write(' '.join(sent) + '\n')

with open('data/eval.txt', 'w') as f:
  for i in tqdm(range(10000)):
    sent = []
    sent.append(np.random.choice(a = np.arange(unigram.shape[0]), p = unigram))
    for pos in range(2, np.random.randint(low = 100, high = 120)): 
      row_sum = bigram[sent[-1]].sum()
      if row_sum > 0:
        sent.append(np.random.choice(a = np.arange(unigram.shape[0]), p = bigram[sent[-1]]))
      else:
        sent.append(np.random.choice(a = np.arange(unigram.shape[0]), p = unigram))
    sent = [str(order_to_token[x]) if x != 3000 else '3' for x in sent]
    f.write(' '.join(sent) + '\n')
