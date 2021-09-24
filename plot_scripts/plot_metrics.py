import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
import sys


######## Settings ########
max_los = 0.3 # 1.2
latex_export = False
line_style = ["-", ":"]
color = ["steelblue", "coral", "C2", "C3"]
lw = 1

path_list = ["results/julian/unet_binary/",
                    "results/julian/unet_jaccard/"]
output_path = "plots/losses/unet_jaccard_binary_metrics.pgf"
label = ['Binary', 'Jaccard']
loc1 = 3
loc2 = 1

if latex_export:
    import matplotlib
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'font.size': 10,
        'text.usetex': True,
        'pgf.rcfonts': False,
    })


 ######## Plot #########

fig, ax = plt.subplots()
fig.set_size_inches(w=5, h=3)
ax.set_xlabel(r'Epochs', fontsize=10)
ax.set_ylabel(r'Loss', fontsize=10)
custom_lines = []
i=0
for path in path_list:
    with open(path + "/history.json") as f:
        hist =  json.load(f)
    loss_list = list(hist["dice_metric"].values())
    val_loss_list = list(hist["val_dice_metric"].values())
    try:
        max_epochs
    except NameError:
        max_epochs= None
    if max_epochs is not None:
        loss_list = loss_list[:max_epochs]
        val_loss_list = val_loss_list[:max_epochs]
    ax.plot(loss_list, lw=lw, color=color[i], ls = line_style[0])
    ax.plot(val_loss_list, lw=lw, color=color[i], ls = line_style[1])
    custom_lines.append(Line2D([0], [0], color=color[i], lw=lw))
    i+=1

legend1 = plt.legend(custom_lines, label, loc = loc1)

custom_lines = [Line2D([0], [0], color="black", lw=lw, ls = line_style[0]),
                Line2D([0], [0], color="black", lw=lw, ls = line_style[1])]
legend2 = plt.legend(custom_lines, ["dice_metric", "val_dice_metric"],loc=loc2)
ax.set_ylim(0,max_los)
try:
    max_epochs
except NameError:
    max_epochs= None
if max_epochs is not None:
    ax.set_xlim(0,max_epochs-1)
ax.add_artist(legend1)
ax.add_artist(legend2)
ax.grid()

if latex_export:
    plt.savefig(output_path,bbox_inches='tight')
else:
    plt.show()