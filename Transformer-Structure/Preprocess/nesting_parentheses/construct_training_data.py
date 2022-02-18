#!/usr/bin/env python
import numpy as np
x = np.load("temp.npy")
x = x.astype(str)
cursor = 0
with open("temp.txt", 'a+') as f:
  for count in range(235000):
    l = np.random.randint(low = 80, high = 125)  
    f.write(" ".join(x[cursor: cursor + l]) + "\n")
    cursor += l
