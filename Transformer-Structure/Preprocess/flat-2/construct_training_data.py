#!/usr/bin/env python
import numpy as np
x = np.load("../En/freq.npy")[5:]
x = x/x.sum()
with open("./data/train.txt", 'w') as f:
  for count in range(930000):
    l = np.random.randint(low = 55, high = 60)
    start = np.random.choice(np.arange(x.shape[0]), p = x, size = l)
    data = []
    for y in start:
      for shift in range(2):  
        data.append(str(y + 0))  
    f.write(" ".join(data) + "\n")
with open("./data/eval.txt", 'w') as f:
  for count in range(10000):
    l = np.random.randint(low = 55, high = 60)
    start = np.random.choice(np.arange(x.shape[0]), p = x, size = l)
    data = []
    for y in start:
      for shift in range(2):  
        data.append(str(y + 0))  
    f.write(" ".join(data) + "\n")
