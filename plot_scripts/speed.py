import matplotlib.pyplot as plt

latex_export = False
pdf_export = True

labels_encoder = ["MobileNetV2", "VGG16", "ResNet50V2", "U-Net", "Final-Net"]
speed_encoder = [16.13, 62.94, 72.84, 14.56, 57.77]

labels_unet = ["big images", "sliced images", "sliced images\nnot optimized"]
speed_unet = [2934.48, 14.56, 255.24]
speed2_unet = [2934.48, 14.56*12, 255.24*12]


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

if pdf_export:
        import matplotlib
        matplotlib.rcParams.update({
            'font.family': 'serif',
            #'text.usetex': True,
            'font.size': 12,
                    })


fig, ax = plt.subplots()
fig.set_size_inches(w=5, h=2)
#ax.set_ylabel(r'Speed [ms]', fontsize=10)
ax.set_xlabel(r'Speed [ms]', fontsize=10)
barlist = ax.barh(labels_encoder, speed_encoder, zorder=3, height=0.3)
barlist[1].set_color("C1")
barlist[2].set_color("C2")
barlist[3].set_color("C3")
ax.invert_yaxis()
ax.xaxis.grid(zorder=0)
if latex_export:
    plt.savefig("plots/speed/encoder.pgf",bbox_inches='tight')
if pdf_export:
    plt.savefig("plots/speed/encoder.svg" + ".svg",bbox_inches='tight', transparent=True)
else:
    plt.show()

fig, ax = plt.subplots()
fig.set_size_inches(w=5, h=1.5)
ax.set_xlabel(r'Speed [ms]', fontsize=10)
#ax.set_ylabel(r'Speed [ms]', fontsize=10)
ax.set_title("U-Net")
ax.barh(labels_unet, speed_unet, zorder=3, height=0.3, label = "one image")
ax.barh(labels_unet, speed2_unet, zorder=2, height=0.3, label = "twelve images")
ax.xaxis.grid(zorder=0)
ax.legend()
if latex_export:
    plt.savefig("plots/speed/unet.pgf",bbox_inches='tight')
if pdf_export:
    plt.savefig("plots/speed/unet.svg" + ".svg",bbox_inches='tight', transparent=True)
else:
    plt.show()