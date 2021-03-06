import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np

vmax = 1 # 1 ####single(1) oder multiclass(2)
figure_width = 5.5
latex_export = False
output_path = "plots/predictions/advanced_decoder/SC/MNV2_MC_AddedLayer2.pgf"

img_path = "plots/predictions/advanced_decoder/SC"

nets = ["MobileNetV2_SC_AddDropout","MobileNetV2_SC_AddTranspose", "MobileNetV2_SC_advanced_decoder_2"]
imgs = [0,1,2]


titles = ["Image", "Mask"] + ["Dropout","Transpose","Advanced"]


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



if latex_export:
    plt.savefig(output_path,bbox_inches='tight')
else:
    plt.show()
