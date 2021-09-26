import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np

figure_width = 5.5
latex_export = False
pdf_export = True
test = latex_export and pdf_export
assert(not test)


imgs = [0,1,2]
plot_types = ["Encoder_sC", "Encoder_mC", "AdvancedDecoder_SC", "AdvancedDecoder_MC", "BigUnet", "UnetJaccard"]
plot_types = ["BigUnet_mC"]
#plot_types = ["AdvancedDecoder_SC", "AdvancedDecoder_MC"]

for plot_type in plot_types:

    if plot_type == "Encoder_sC":
        vmax = [1,1,1,1]
        output_path = "plots/predictions/encoder_sC/Prediction_encoder_sC"
        img_path = "plots/predictions/encoder_sC"
        nets = ["MNV2", "VGG16", "ResNet50V2"]
        titles = ["Image", "Mask"] + ["MobileNetV2", "VGG16", "Reset50V2"]

    elif plot_type == "Encoder_mC":
        vmax = [2,2,2,2]
        output_path = "plots/predictions/encoder_mC/Prediction_encoder_MC"
        img_path = "plots/predictions/encoder_mC"
        nets = ["MNV2", "VGG16", "ResNet50V2"]
        titles = ["Image", "Mask"] + ["MobileNetV2", "VGG16", "Reset50V2"]

    elif plot_type == "AdvancedDecoder_MC":
        vmax = [2,2,2,2]
        output_path = "plots/predictions/advanced_decoder/MC/MNV2_MC_AddedLayer2"
        img_path = "plots/predictions/advanced_decoder/MC"
        nets = ["MobileNetV2_MC_AddDropout", "MobileNetV2_MC_AddTranspose", "MobileNetV2_MC_advanced_decoder_2"]
        titles = ["Image", "Mask"] + ["Dropout", "Transpose", "Advanced"]
    
    elif plot_type == "AdvancedDecoder_SC":
        vmax = [1,1,1,1]
        output_path = "plots/predictions/advanced_decoder/SC/MNV2_SC_AddedLayer2"
        img_path = "plots/predictions/advanced_decoder/SC"
        nets = ["MobileNetV2_SC_AddDropout", "MobileNetV2_SC_AddTranspose", "MobileNetV2_SC_advanced_decoder_2"]
        titles = ["Image", "Mask"] + ["Dropout", "Transpose", "Advanced"]
    
    elif plot_type == "BigUnet_mC":
        vmax = [2,2,2,2]
        output_path = "plots/predictions/unet_sliced_big_mC"
        img_path = "plots/predictions/unet/plot2"
        nets = ["Big_images_mC", "Sliced_images"]
        titles = ["Image", "Mask"] + ["big\nimages", "sliced\nimages"]
        horizontal = [3,7,1]
    
    elif plot_type == "UnetJaccard":
        vmax = [1,1,1,1]
        output_path = "plots/predictions/unet_jaccard"
        img_path = "plots/predictions/unet/plot3_jaccard"
        nets = ["jaccard", "binary"]
        titles = ["Image", "Mask"] + ["Jaccard \n Distance", "Binary \n Crossentropy"]
    elif plot_type == "FinalNet":
        vmax = [2,1,1,2,2]
        output_path = "plots/predictions/final_net"
        img_path = "plots/predictions/unet/plot4_mobile_unet"
        nets = ["unet_mobilenet_sC", "unet_mobilenet_sC_fine_tuning", "unet_mobilenet_mC", "unet_mobilenet_mC_fine_tuning"]
        titles = ["Image", "Mask"] + ["SC", "SC with \n fine tuning", "MC", "MC with \n fine tuning"]





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
            axes[(i)*len(titles)+1].imshow(mask, vmin = 0, vmax = vmax[0])
            axes[(i)*len(titles)+1].axis("off")
        j = 2
        for net in nets:
            with open(img_path + f"/{net}_pred_{i}.npy", "rb") as f:
                pred = np.load(f)
                if net == "Big_images" or net == "Big_images_mC":
                    pred = np.split(pred, 12, axis=1)[horizontal[i]]
                axes[(i)*len(titles)+j].imshow(pred, vmin = 0, vmax = vmax[j-1])
                axes[(i)*len(titles)+j].axis("off")
                j+=1



    if pdf_export:
        plt.savefig(output_path + ".svg",bbox_inches='tight', transparent=True)
    if latex_export:
        plt.savefig(output_path + ".pgf",bbox_inches='tight')
    if not latex_export and not pdf_export:
        plt.show()
