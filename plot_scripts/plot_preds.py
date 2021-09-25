import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np

vmax = 2 # 1 ####single(1) oder multiclass(2)
figure_width = 5.5
latex_export = False
pdf_export = True
output_path = "plots/predictions/encoder_mC/Prediction_encoder_MC"
#output_path = "plots/predictions/encoder_sC/Prediction_encoder_sC"

img_path = "plots/predictions/encoder_mC"
#img_path = "plots/predictions/encoder_sC"
nets = ["MNV2", "VGG16", "ResNet50V2"]
imgs = [0,1,2]


titles = ["Image", "Mask"] + ["MobileNetV2", "VGG16", "Reset50V2"]


if latex_export:
    import matplotlib
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'font.size': 10,
        'text.usetex': True,
        'pgf.rcfonts': False
    })
if pdf_export:
    import matplotlib
    matplotlib.rcParams.update({
        'font.family': 'serif',
        #'text.usetex': True,
        'font.size': 12,
                })

fig, axes = plt.subplots(len(imgs),len(titles), gridspec_kw={'wspace':0, 'hspace':0.1}, squeeze=True)
fig.set_size_inches(w=figure_width, h=figure_width/len(titles)*len(imgs))
axes = axes.flatten()

for i in range(len(titles)):
    axes[i].set_title(titles[i], fontsize=10)

for i in imgs:
    with open(img_path + f"/img_{i}.npy", "rb") as f:
        img = np.load(f)
        axes[i*len(titles)].imshow(img)
        axes[i*len(titles)].axis("off")
    with open(img_path + f"/mask_{i}.npy", "rb") as f:
        mask = np.load(f)
        axes[(i)*len(titles)+1].imshow(mask, vmin = 0, vmax = vmax)
        axes[(i)*len(titles)+1].axis("off")
    j = 2
    for net in nets:
        with open(img_path + f"/{net}_pred_{i}.npy", "rb") as f:
            pred = np.load(f)
            axes[(i)*len(titles)+j].imshow(pred, vmin = 0, vmax = vmax)
            axes[(i)*len(titles)+j].axis("off")
            j+=1



if pdf_export:
    plt.savefig(output_path + ".svg",bbox_inches='tight', transparent=True)
if latex_export:
    plt.savefig(output_path + ".pgf",bbox_inches='tight')
if not latex_export and not pdf_export:
    plt.show()
