#!/usr/bin/env python
import numpy as np
x = np.load("../En/freq.npy")[5:]
x = x/x.sum()

with open("./data/eval.txt", 'w') as f:
  for count in range(10000):
    l = np.random.randint(low = 55, high = 62)
    tokens = np.random.choice(np.arange(x.shape[0]), p = x, size = l)
    data = []
    while len(tokens) > 2:
      i = np.random.randint(low = 2, high = 3)
      subarray = tokens[:i]
      tokens = tokens[i:]
      subarray = np.concatenate((subarray, subarray))
      np.random.shuffle(subarray)
      data.extend([str(x) for x in subarray])  
    f.write(" ".join(data) + "\n")
with open("./data/train.txt", 'w') as f:
  for count in range(930000):
    l = np.random.randint(low = 55, high = 62)
    tokens = np.random.choice(np.arange(x.shape[0]), p = x, size = l)
    data = []
    while len(tokens) > 2:
      i = np.random.randint(low = 2, high = 3)
      subarray = tokens[:i]
      tokens = tokens[i:]
      subarray = np.concatenate((subarray, subarray))
      np.random.shuffle(subarray)
      data.extend([str(x) for x in subarray])  
    f.write(" ".join(data) + "\n")

