#!/usr/bin/env python
import numpy as np
x = np.load("../En/freq.npy")[5:]
x = np.ones_like(x)
freq = x / x.sum()
with open("./data/train.txt", "w") as f:
  for i in range(930000):
    length = np.random.randint(low = 100, high = 120)
    sentence = np.random.choice(len(freq), length, p = freq)
    sentence = " ".join(list(sentence.astype(str)))
    f.write(sentence + "\n") 
with open("./data/eval.txt", "w") as f:
  for i in range(10000):
    length = np.random.randint(low = 100, high = 120)
    sentence = np.random.choice(len(freq), length, p = freq)
    sentence = " ".join(list(sentence.astype(str)))
    f.write(sentence + "\n") 
     
