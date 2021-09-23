import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
import sys

######## Settings ########
max_los = 0.3 # 1.2
latex_export = True
line_style = ["-", ":"]
color = ["steelblue", "coral", "C2", "C3"]
lw = 1

##plot all
#plot_types = ["EncoderA", "EncoderB", "EncoderC", "EncoderD", "UnetA", "UnetB", "UnetC", "UnetD", "DecoderA", "DecoderB", "DecoderC", "DecoderD", "DecoderE"]
##plot single
plot_types = ["DecoderA", "DecoderB", "DecoderC", "DecoderD", "DecoderE"]
#plot_types = ["EncoderD"] #A, B, C or D
#plot_type = ["UnetD"] #A, B, C or D
#plot_types = ["DecoderE"] #A, B, C or D


for plot_type in plot_types:
    if plot_type == "EncoderA":
        path_list = ["results/lena/mobilenetV2/multiLabel/standard",
                    "results/julian/vgg16_1",
                    "results/samuel/ResNet50V2_wS_SC_ES_50epochs"]
        output_path = "plots/losses/encoder_wS_SC_ES_50epochs.pgf"
        label = ['MobileNetV2', 'VGG16', 'ResNet50V2']
        loc1 = 3
        loc2 = 1

    elif plot_type == "EncoderB":
        path_list = ["results/lena/mobilenetV2/oneLabel/standard_V2",
                        "results/julian/vgg16_3",
                        "results/samuel/ResNet50V2_wS_MC_ES_50epochs"]
        output_path = "plots/losses/encoder_wS_MC_ES_50epochs.pgf"
        label = ['MobileNetV2', 'VGG16', 'ResNet50V2']
        loc1 = 3
        loc2 = 1

    elif plot_type == "EncoderC":
        path_list = ["results/lena/mobilenetV2/oneLabel/withoutShift/noEarlyStopping/epochs100_validationSplit0-1",
                        "results/julian/vgg16_7",
                        "results/samuel/ResNet50V2_nS_SC_100epochs"]
        output_path = "plots/losses/encoder_nS_SC_100epochs.pgf"
        label = ['MobileNetV2', 'VGG16', 'ResNet50V2']
        loc1 = 1
        loc2 = 7

    elif plot_type == "EncoderD":
        path_list = ["results/lena/mobilenetV2/multiLabel/withoutShift/noEarlyStopping_epochs100_validationSplit0-1",
                        "results/julian/vgg16_6",
                        "results/samuel/ResNet50V2_nS_MC_100epochs"]
        output_path = "plots/losses/encoder_nS_MC_100epochs.pgf"
        label = ['MobileNetV2', 'VGG16', 'ResNet50V2']
        loc1 = 1
        loc2 = 7

    elif plot_type == "UnetA":
        path_list = ["results/julian/unet_1",
                        "results/julian/unet_3",
                        "results/julian/unet_2"]
        output_path = "plots/losses/unet_wS_SC_50epochs.pgf"
        label = ['Complexity = 2', 'Complexity = 4', 'Complexity = 6']
        loc1 = 3
        loc2 = 4

    elif plot_type == "UnetB":
        path_list = ["results/julian/unet_3",
                        "results/julian/unet_4"]
        output_path = "plots/losses/unet_wS_nS_SC_50epochs.pgf"
        label = ['with shift', 'whitout shift']
        loc1 = 3
        loc2 = 4

    elif plot_type == "UnetC":
        path_list = ["results/julian/unet_256x3072_2",
                        "results/julian/unet_256x3072",
                        "results/julian/unet_256x3072_3"]
        output_path = "plots/losses/unet_nS_SC_20epochs_bigimg.pgf"
        label = ['Complexity = 3', 'Complexity = 4', 'Complexity = 5']
        loc1 = 1
        loc2 = 3

    elif plot_type == "UnetD":
        path_list = ["results/julian/unet_5",
                        "results/julian/unet_256x3072_4"]
        output_path = "plots/losses/unet_wS_nS_MC_50epochs.pgf"
        label = ['sliced images', 'big images']
        loc1 = 3
        loc2 = 1

    elif plot_type == "DecoderA":
        path_list = ["results/samuel/MobileNetV2_MC",
                        "results/samuel/MobileNetV2_MC_BN"]
        output_path = "plots/losses/Decoder_nS_MC_BN_50epochs.pgf"
        label = ['without BN', 'with BN']
        loc1 = 1
        loc2 = 3
    
    elif plot_type == "DecoderB":
        path_list = ["results/samuel/MobileNetV2_MC_2xConv2D",
                        "results/samuel/MobileNetV2_MC_BN_2xConv2D"]
        output_path = "plots/losses/Decoder_nS_MC_BN_2xConv2D_50epochs.pgf"
        label = ['without BN', 'with BN']
        loc1 = 1
        loc2 = 3

    elif plot_type == "DecoderC":
        path_list = ["results/samuel/MobileNetV2_MC_Conv2DTranspose",
                        "results/samuel/MobileNetV2_MC_BN_Conv2DTranspose",
                        "results/samuel/MobileNetV2_SC_BN_Conv2DTranspose"]
        output_path = "plots/losses/Decoder_nS_MCSC_BN_Conv2DTranspose_50epochs.pgf"
        label = ['Multi-Class without BN', 'Multi-Class with BN', 'Single-Class with BN']
        loc1 = 1
        loc2 = 3


    elif plot_type == "DecoderD":
        path_list = ["results/samuel/MobileNetV2_MC_AddUpsampling",
                        "results/samuel/MobileNetV2_MC_AddTranspose",
                        "results/samuel/MobileNetV2_MC_AddDropout",
                        "results/samuel/MobileNetV2_MC_KerasModel"]
        output_path = "plots/losses/Decoder_nS_MC_AddedLayer_50epochs.pgf"
        label = ['Add Upsampling', 'Add Conv2DTranspose', 'Add Dropout', 'Keras Model']
        loc1 = 1
        loc2 = 3

    elif plot_type == "DecoderE":
        max_epochs = 25
        path_list = ["results/samuel/MobileNetV2_SC_AddUpsampling",
                        "results/samuel/MobileNetV2_SC_AddTranspose",
                        "results/samuel/MobileNetV2_SC_AddDropout",
                        "results/samuel/MobileNetV2_SC_KerasModel"]
        output_path = "plots/losses/Decoder_nS_SC_AddedLayer_50epochs.pgf"
        label = ['Add Upsampling', 'Add Conv2DTranspose', 'Add Dropout', 'Keras Model']
        loc1 = 1
        loc2 = 3

    elif plot_type == "Segnet":
        path_list = ["results/lena/segnet02/OC_nS_vs01", 
                        "results/lena/segnet02/MC_wS_vs01"]
        output_path = "plots/losses/segnet_nS_SC_MC_50epochs.pgf"
        label = ["Single-Class", "Multi-Class"]
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
    fig.set_size_inches(w=5, h=3.5)
    ax.set_xlabel(r'Epochs', fontsize=10)
    ax.set_ylabel(r'Loss', fontsize=10)
    custom_lines = []
    i=0
    for path in path_list:
        with open(path + "/history.json") as f:
            hist =  json.load(f)
        loss_list = list(hist["loss"].values())
        val_loss_list = list(hist["val_loss"].values())
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
    legend2 = plt.legend(custom_lines, ["loss", "val_loss"],loc=loc2)
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

    # from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    # from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    # axins = zoomed_inset_axes(ax, 3, loc=9) # zoom = 6
    # i=0
    # for path in path_list:
    #     with open(path + "/history.json") as f:
    #         hist =  json.load(f)

    #     axins.plot(list(hist["loss"].values()), lw=lw, color=color[i], ls = line_style[0])
    #     axins.plot(list(hist["val_loss"].values()), lw=lw, color=color[i], ls = line_style[1])
    #     i+=1

    # x1, x2, y1, y2 = 85, 100, 0.03, 0.08
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    # axins.grid()
    # plt.xticks(visible=False)
    # plt.yticks(visible=False)

    # mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.3")
    if latex_export:
        plt.savefig(output_path,bbox_inches='tight')
    else:
        plt.show()

