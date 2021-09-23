import matplotlib.pyplot as plt
from matplotlib import image

img_paths = []

fig, axes = plt.subplots((2,5))
axes = axes.flatten()
titles = ["Image", "Mask", "Prediction MoibleNetV2", "Prediction VGG16", "Prediction ResNet"]

for i in range(len(titles)):
    axes[i].set_title(titles[i])

for i in range(len(img_paths)):
    img = image.imread(img_paths[i])
    axes[i].imshow(img)
    axes[i].axis("off")
