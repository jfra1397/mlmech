import json
import matplotlib.pyplot as plt

paths = [['results\samuel\MobileNetV2\history.json','results\samuel\MobileNetV2_BN\history.json'],
        ['results\samuel\MobileNetV2_2xConv2D\history.json','results\samuel\MobileNetV2_BN_2xConv2D\history.json'],
        ['results\samuel\MobileNetV2_Conv2DTranspose\history.json','results\samuel\MobileNetV2_BN_Conv2DTranspose\history.json'],
        ['results\samuel\MobileNetV2_AddUpsampling\history.json','results\samuel\MobileNetV2_AddTranspose\history.json'],
        ['results\samuel\MobileNetV2_AddDropout\history.json','results\samuel\MobileNetV2_KerasModel\history.json']
]
titles = [['Comparison with & without Batch \n Normalization for a Mobile Net V2'],
            ['Comparison With & Without Batch \nNormalization for a Mobile Net V2 \nwith two Convolutional Layers'],
            ['Comparison With & Without Batch \n Normalization for a Mobile Net V2 \nwith a Convolutional Transpose Layer'],
            ['Comparison of Added layers with \n an Upsampling & with a Transpose Layers instead'],
            ['Comparison of Added layers with \n with Dropout applied to a \n Modelapproach from Keras.io']
]
losslabels = [['Loss -without BN', 'Loss -with BN'],
            ['Loss -without BN', 'Loss -with BN'],
            ['Loss -without BN', 'Loss -with BN'],
            ['Loss -with UpSampling', 'Loss -with Transpose'],
            ['Loss -with Dropout', 'Loss -similiar to Keras-example']
]
val_losslabels = [['Val_Loss -without BN', 'Val.Loss -with BN'],
            ['Val_Loss -without BN', 'Val.Loss -with BN'],
            ['Val_Loss -without BN', 'Val.Loss -with BN'],
            ['Val_Loss -withUpsampling', 'Val.Loss -with Transpose'],
            ['Val_Loss -with Dropout', 'Val.Loss -similiar to Keras-example']
]
savetitles = [["results\samuel\MobileNetV2_BN_comparison-50epochs.png", "results\samuel\MobileNetV2_BN_comparison-50epochs.pdf"],
            ["results\samuel\MobileNetV2_BN_2xConv2D_comparison-50epochs.png","results\samuel\MobileNetV2_BN_2xConv2D_comparison-50epochs.pdf"],
            ["results\samuel\MobileNetV2_BN_Conv2DTranspose_comparison-50epochs.png","results\samuel\MobileNetV2_BN_Conv2DTranspose_comparison-50epochs.pdf"],
            ["results\samuel\MobileNetV2_AddUpsampligTranspose-50epochs.png", "results\samuel\MobileNetV2_AddUpsampligTranspose-50epochs.pdf"],
            ["results\samuel\MobileNetV2_AddDropoutKerasModel-50epochs.png", "results\samuel\MobileNetV2_AddDropoutKerasModel-50epochs.pdf"]
]
max_loss = 0.25
max_epochs = 50
for i in range(0,len(paths)):
    with open(paths[i][0]) as f:
        hist1 =  json.load(f)
    with open(paths[i][1]) as f:
        hist2 =  json.load(f)

    fig, ax = plt.subplots()
    plt.plot(hist1["loss"].values(), lw=4, color='steelblue', label=losslabels[i][0])
    plt.plot(hist1["val_loss"].values(), lw=3, color='steelblue', linestyle='dotted', label=val_losslabels[i][0])
    plt.plot(hist2["loss"].values(), lw=4, color='coral', label=losslabels[i][1])
    plt.plot(hist2["val_loss"].values(), lw=3, color='coral', linestyle='dotted', label=val_losslabels[i][1])
    ax.set_xlim(0,max_epochs)
    ax.set_xlabel('Epochs', fontsize=15)
    ax.set_ylim(0,max_loss)
    ax.set_ylabel('Loss', fontsize=15)
    ax.set_title(titles[i][0])
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.show()
    fig.savefig(savetitles[i][0])
    fig.savefig(savetitles[i][1])
    plt.close()
