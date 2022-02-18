#!/usr/bin/env python
import numpy as np
x = np.load("../En/freq.npy")[5:]
x = x/x.sum()
with open("data/train.txt", 'w') as f:
  for count in range(930000):
    l = np.random.randint(low = 50, high = 62)
    data = np.random.choice(a = np.arange(x.shape[0]), p = x, size = l)
    data = np.concatenate((data, data))
    np.random.shuffle(data)
    data = [str(y) for y in data]
    f.write(" ".join(data) + "\n")


with open("data/eval.txt", 'w') as f:
  for count in range(10000):
    l = np.random.randint(low = 50, high = 62)
    data = np.random.choice(a = np.arange(x.shape[0]), p = x, size = l)
    data = np.concatenate((data, data))
    np.random.shuffle(data)
    data = [str(y) for y in data]
    f.write(" ".join(data) + "\n")
