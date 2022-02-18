import argparse
from collections import deque
import hashlib
import numpy as np
import torch

import os
import sys
num_tokens = 29995
freq = np.load("../En/freq.npy")[5:]
freq = np.array(freq)
freq = freq[:num_tokens]
ps = freq / sum(freq)
word_indices = np.arange(len(freq))
open_prob = 0.4
print("Loaded and initialized everything")

data = 128 * 235000
result = []
print("Sampling open chars")
open_deque = deque()
open_decision = np.random.choice([0, 1], data, p = [1 - open_prob, open_prob])
samples = np.random.choice(word_indices, data, p=ps)
print("Finished sampling, starting construction")
for i in range(data):
    if i % 1000000 == 0:
        print(f"i is {i}")
    if open_decision[i] or len(open_deque) == 0:
        result.append(samples[i])
        open_deque.append(samples[i])
    else:
        last_open = open_deque.pop()
        result.append(last_open)
np.save("temp.npy", result)

