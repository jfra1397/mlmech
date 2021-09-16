path = "results/"

import os
import json
import math
import numpy as np

import sys


results = {}
for root, dirs, files in os.walk(path):
    for file in files:
        if file == ("history.json"):
            with open(os.path.join(root, file)) as f:
                key =  os.path.join(root, file).replace(path,'')
                results[key] = json.load(f)

import matplotlib.pyplot as plt

all_loss = []
all_epochs = []
# for _, hist in results.items():
#     all_loss += list(hist["loss"].values())
#     all_loss += list(hist["val_loss"].values())
#     all_epochs += list(hist["val_loss"].keys())

# all_loss = np.array(all_loss)
# all_epochs = np.array([float(e) for e in all_epochs])
max_loss = 0.3 # all_loss.max()
max_epochs = 20 # all_epochs.max()


fig, axes = plt.subplots(math.ceil(len(results)/3),3)

axes = axes.flatten()
i = 0
for name, hist in results.items():
    axes[i].plot(hist["loss"].values(), lw=4, label="loss")
    axes[i].plot(hist["val_loss"].values(), lw=4, label="val_loss")
    axes[i].set_title(name)
    axes[i].set_ylim(0,max_loss)
    axes[i].set_xlim(0,max_epochs)
    i +=1

plt.show()