import sys
import json

import matplotlib.pyplot as plt

path  = sys.argv[1]


with open(path) as f:
    hist =  json.load(f)

fig, ax = plt.subplots()
plt.plot(hist["loss"].values(), lw=4, label="loss")
plt.plot(hist["val_loss"].values(), lw=4, label="val_loss")

ax.set_xlabel(r'$Epochs$', fontsize=15)
ax.set_ylabel(r'$Loss$', fontsize=15)
ax.set_title(path)

ax.grid(True)
fig.tight_layout()
# ax.set(xlabel='Epochs', ylabel='Loss', title=path)
# ax.grid()
fig.savefig(path +".png")
fig.savefig(path +".pdf")
plt.show()