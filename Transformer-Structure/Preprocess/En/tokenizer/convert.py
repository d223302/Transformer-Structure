#!/usr/bin/env python3
import json
f = open("vocab.json", encoding='utf-8')
x = json.load(f)
y = {}
for k, v in x.items():
  if k == "<mask>":
    y[k] = 4
    print(k)
  if v <=3:
    y[k] = v
  else:
    y[k] = v + 1
with open("new_vocab.json", encoding='utf-8', mode = 'w') as h:
  json.dump(y, h, ensure_ascii=False)
