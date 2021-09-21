import json
import matplotlib.pyplot as plt

paths = [['results\samuel\ResNet50V2_nS_MC_100epochs\history.json'],
        ['results\samuel\ResNet50V2_nS_SC_100epochs\history.json'],
        ['results\samuel\ResNet50V2_wS_MC_50epochs\history.json'],
        ['results\samuel\ResNet50V2_wS_SC_50epochs\history.json'],
        ['results\samuel\ResNet50V2_bB_MC\history.json'],
        ['results\samuel\ResNet50V2_bB_SC2\history.json']
]
titles = [['ResNet50V2 without a shift in trainings data \n and Multi-Class over 100 epochs'],
            ['ResNet50V2 without a shift in trainings data \n and Single-Class over 100 epochs'],
            ['ResNet50V2 with a shift in trainings data \n and Multi-Class over 50 epochs'],
            ['ResNet50V2 with a shift in trainings data \n and Single-Class over 50 epochs'],
            ['ResNet50V2 with a train batch of 64 images and Multi-Class'],
            ['ResNet50V2 with a train batch of 64 images and Single-Class']
]
losslabels = [['Loss'],
            ['Loss'],
            ['Loss'],
            ['Loss'],
            ['Loss'],
            ['Loss']            
]
val_losslabels = [['Val_Loss'],
                    ['Val_Loss'],
                    ['Val_Loss'],
                    ['Val_Loss'],
                    ['Val_Loss'],
                    ['Val_Loss']            
]
savetitles = [['results\samuel\ResNet50V2_nS_MC_100epochs.pdf','results\samuel\ResNet50V2_nS_MC_100epochs.png'],
        ['results\samuel\ResNet50V2_nS_SC_100epochs.pdf','results\samuel\ResNet50V2_nS_SC_100epochs.png'],
        ['results\samuel\ResNet50V2_wS_MC_50epochs.pdf','results\samuel\ResNet50V2_wS_MC_50epochs.png'],
        ['results\samuel\ResNet50V2_wS_SC_50epochs.pdf','results\samuel\ResNet50V2_wS_SC_50epochs.png'],
        ['results\samuel\ResNet50V2_bB_MC.pdf','results\samuel\ResNet50V2_bB_MC.png'],
        ['results\samuel\ResNet50V2_bB_SC.pdf','results\samuel\ResNet50V2_bB_SC.png']
]

for i in range(0,len(paths)):
    with open(paths[i][0]) as f:
        hist1 =  json.load(f)

    fig, ax = plt.subplots()
    plt.plot(hist1["loss"].values(), lw=4, color='steelblue', label=losslabels[i][0])
    plt.plot(hist1["val_loss"].values(), lw=4, color='coral', label=val_losslabels[i][0])
    ax.set_xlabel('Epochs', fontsize=15)
    ax.set_ylabel('Loss', fontsize=15)
    ax.set_title(titles[i][0])
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.show()
    fig.savefig(savetitles[i][0])
    fig.savefig(savetitles[i][1])
    plt.close()
