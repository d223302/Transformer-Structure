#!/usr/bin/env python
import numpy as np
x = np.load("../En/freq.npy")[5:-4]
x = x/x.sum()

with open("./data/eval.txt", 'w') as f:
  for count in range(10000):
    l = np.random.randint(low = 14, high = 15)
    start = np.random.choice(np.arange(x.shape[0]), p = x, size = l)
    data = []
    for y in start:
      sub_array = np.arange(y, y + 8)
      np.random.shuffle(sub_array)
      data.extend([str(token) for token in sub_array])   
    f.write(" ".join(data) + "\n")

with open("./data/train.txt", 'w') as f:
  for count in range(930000):
    l = np.random.randint(low = 14, high = 15)
    start = np.random.choice(np.arange(x.shape[0]), p = x, size = l)
    data = []
    for y in start:
      sub_array = np.arange(y, y + 8)
      np.random.shuffle(sub_array)
      data.extend([str(token) for token in sub_array])  
    f.write(" ".join(data) + "\n")
