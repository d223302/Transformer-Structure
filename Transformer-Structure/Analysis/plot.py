#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
plt.locator_params(axis='y', nbins=10)
sns.set()
data_map = {'Zipf_baseline': ['Uni-gram', sns.color_palette("Paired")[1]],
            'Random_baseline': ['Uniform', sns.color_palette("Paired")[1]],
            'Kannada': ['Kannada', sns.color_palette("Paired")[4]],
            'local-4': ['Shuffle-4', sns.color_palette("Paired")[2]],
            'repeat-4': ['Repeat 4', sns.color_palette("Paired")[2]],
            'Bigram': ['Bi-gram', sns.color_palette("Paired")[1]], 
            'global-infinite': ['Shuffle', sns.color_palette("Paired")[6]], 
            'shuffle-sort': ['Sort Shuffle', sns.color_palette("Paired")[6]], 
            'nesting_parentheses': ['Nesting Parentheses', sns.color_palette("Paired")[8]],
            'flat_parentheses': ['Flat Parentheses', sns.color_palette("Paired")[8]],
            'local_flat': ['Flat Parentheses-6', sns.color_palette("Paired")[8]],
            'local_flat_4': ['Flat Parentheses-4', sns.color_palette("Paired")[8]]
            }
fig, axs = plt.subplots(3, 2, sharex = True, sharey = False, figsize = (16, 8))
fig.tight_layout(pad=2.0, w_pad=0.5, h_pad=1.5)
#fig.suptitle('')
data_count = 0
with open('result.txt', 'r') as f:
  while True:
    data_type = f.readline()
    if data_type == "":
      break
    data_type = data_type.strip()
    data = f.readline().strip('\t').split()
    data = [float(x) for x in data]
    conf = f.readline().strip()[:6]
    entropy = f.readline().strip()[:6]
    if data_type == "Bigram_En" or data_type == "flat_parentheses":
      continue

    #axs[data_count].text(0.02, 0.96, 'Entropy: ' + entropy + '\nConfidence: ' + conf, 
    #  horizontalalignment='left', verticalalignment='top', 
    #  transform=axs[data_count].transAxes)
    print(data_type)
    axs[data_count % 3, data_count // 3].bar(np.arange(81) - 40,
                                             data, width = 1.0,
                                             color = data_map[data_type][1])
    axs[data_count % 3, data_count // 3].set_title(data_map[data_type][0],
                                                   fontsize = 16)
    data_count += 1
plt.savefig('attn_dis.pdf')
