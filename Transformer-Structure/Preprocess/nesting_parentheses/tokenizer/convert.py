#!/usr/bin/env python3
with open("vocab.txt", "a+") as f:
  for i in range(29986):
    f.write(str(i) + "\n")
