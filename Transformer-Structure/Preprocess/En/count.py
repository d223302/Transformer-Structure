#!/usr/bin/env python3
import transformers
import numpy as np
x = np.zeros(30000)
with open("data/train_tokenized.txt", 'r') as f:
  while True:
    line = f.readline()
    if line == "":
      break
    line = line.rstrip()
    for idx in line.split():
      x[int(idx)] += 1
np.save("freq.npy", x)
      
