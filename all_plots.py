from absl.logging import ERROR
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
import sys

nets = ['MobileNetV2', 'VGG16', 'ResNet']
line_style = ["-", "-.", "-."]
color = ["steelblue", "coral", "C2"]
lw = 1

max_los = 0.3

######## A #########
if sys.argv[1] == "A":
    title = "Blub"
    path_list = ["results/lena/mobilenetV2/multiLabel/standard",
                "results/julian/vgg16_1",
                "results/samuel/ResNet50V2_wS_SC_ES_50epochs"]
######## B #########
elif sys.argv[1] == "B":
    title = "Blub"
    path_list = ["results/lena/mobilenetV2/oneLabel/standard_V2",
                    "results/julian/vgg16_3",
                    "results/samuel/ResNet50V2_wS_MC_ES_50epochs"]

######## C #########
elif sys.argv[1] == "C":
    title = "Blub"
    path_list = ["results/lena/mobilenetV2/oneLabel/withoutShift/noEarlyStopping/epochs100_validationSplit0-1",
                    "results/julian/vgg16_7",
                    "results/samuel/ResNet50V2_nS_SC_100epochs"]

######## D #########
elif sys.argv[1] == "D":
    title = "Blub"
    path_list = ["results/lena/mobilenetV2/multiLabel/withoutShift/noEarlyStopping_epochs100_validationSplit0-1",
                    "results/julian/vgg16_6",
                    "results/samuel/ResNet50V2_nS_MC_100epochs"]
else:
    raise NameError("Wrong argument!!!")


######## Plot #########

fig, ax = plt.subplots()
ax.set_xlabel(r'$Epochs$', fontsize=15)
ax.set_ylabel(r'$Loss$', fontsize=15)
ax.set_title(title, fontsize=20)
custom_lines = []
i=0
for path in path_list:
    with open(path + "/history.json") as f:
        hist =  json.load(f)

    ax.plot(list(hist["loss"].values()), lw=lw, color=color[i], ls = line_style[0])
    ax.plot(list(hist["val_loss"].values()), lw=lw, color=color[i], ls = line_style[1])
    custom_lines.append(Line2D([0], [0], color=color[i], lw=lw))
    i+=1

legend1 = plt.legend(custom_lines, nets, loc = 1)

custom_lines = [Line2D([0], [0], color="black", lw=lw, ls = line_style[0]),
                Line2D([0], [0], color="black", lw=lw, ls = line_style[1])]
legend2 = plt.legend(custom_lines, ["loss", "val_loss"],loc=3)
ax.set_ylim(0,max_los)
ax.add_artist(legend1)
ax.add_artist(legend2)
ax.grid()

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

axins = zoomed_inset_axes(ax, 3, loc=9) # zoom = 6
i=0
for path in path_list:
    with open(path + "/history.json") as f:
        hist =  json.load(f)

    axins.plot(list(hist["loss"].values()), lw=lw, color=color[i], ls = line_style[0])
    axins.plot(list(hist["val_loss"].values()), lw=lw, color=color[i], ls = line_style[1])
    i+=1

x1, x2, y1, y2 = 85, 100, 0.03, 0.08
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.grid()
plt.xticks(visible=False)
plt.yticks(visible=False)

mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.3")

plt.draw()
plt.show()


