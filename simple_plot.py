import sys
import json

import matplotlib.pyplot as plt

path  = sys.argv[1]


with open(path) as f:
    hist =  json.load(f)


plt.plot(hist["loss"].values(), lw=4, label="loss")
plt.plot(hist["val_loss"].values(), lw=4, label="val_loss")

plt.show()