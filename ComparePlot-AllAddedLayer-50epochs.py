import json
import matplotlib.pyplot as plt

paths = ['results\samuel\MobileNetV2_AddUpsampling\history.json','results\samuel\MobileNetV2_AddTranspose\history.json','results\samuel\MobileNetV2_AddDropout\history.json','results\samuel\MobileNetV2_KerasModel\history.json']
titles = ['Comparison of All Added layers Models']
losslabels = ['Loss -with Upsampling', 'Loss -with Transpose','Loss -with Dropout', 'Loss -similiar to Keras-example']
val_losslabels = ['Val_Loss -with Upsamling', 'Val.Loss -with Transpose','Val_Loss -with Dropout', 'Val.Loss -similiar to Keras-example']
savetitles = ["results\samuel\MobileNetV2_AllAddedLayers-50epochs.png", "results\samuel\MobileNetV2_AllAddedLayers-50epochs.pdf"]
max_loss = 0.15
max_epochs = 50
min_epochs = 0
with open(paths[0]) as f:
    hist1 =  json.load(f)
with open(paths[1]) as f:
    hist2 =  json.load(f)
with open(paths[2]) as f:
    hist3 =  json.load(f)
with open(paths[3]) as f:
    hist4 =  json.load(f)

fig, ax = plt.subplots()
plt.plot(hist1["loss"].values(), lw=2, color='steelblue', label=losslabels[0])
plt.plot(hist1["val_loss"].values(), lw=1.5, color='steelblue', linestyle='dotted', label=val_losslabels[0])
plt.plot(hist2["loss"].values(), lw=2, color='coral', label=losslabels[1])
plt.plot(hist2["val_loss"].values(), lw=1.5, color='coral', linestyle='dotted', label=val_losslabels[1])
plt.plot(hist3["loss"].values(), lw=2, color='forestgreen', label=losslabels[2])
plt.plot(hist3["val_loss"].values(), lw=1.5, color='forestgreen', linestyle='dotted', label=val_losslabels[2])
plt.plot(hist4["loss"].values(), lw=2, color='darkviolet', label=losslabels[3])
plt.plot(hist4["val_loss"].values(), lw=1.5, color='darkviolet', linestyle='dotted', label=val_losslabels[3])
ax.set_xlim(min_epochs,max_epochs)
ax.set_xlabel('Epochs', fontsize=15)
ax.set_ylim(0,max_loss)
ax.set_ylabel('Loss', fontsize=15)
ax.set_title(titles[0])
ax.legend(loc='lower left')
#ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
ax.grid(True)
fig.tight_layout()
plt.show()
fig.savefig(savetitles[0])
fig.savefig(savetitles[1])
plt.close()
