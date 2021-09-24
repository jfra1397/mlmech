import matplotlib.pyplot as plt
from os import path
from load_data import CustomDataGenerator

#DATA GENERATOR
img_dir = "images/"
mask_dir = "labels/"
image_extension = ".png"
mask_extension = ".png"
img_size = (256, 256, 3)
batch_size = 16
horizontal_split = 1#2 #1
vertical_split = 1

seed = 42
onelabel = True
shift = False
single_img = False

#PREPROCESS FUNCTION OF THE PRETRAINED ENCODER
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess
#from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess
#from tensorflow.python.keras.applications.vgg16 import preprocess_input as preprocess
preprocess_fcn = preprocess

import numpy as np
train, validation = CustomDataGenerator.generate_data(batch_size, img_dir, mask_dir,
                                                        horizontal_split, vertical_split, image_extension, mask_extension, 
                                                        preprocess_fcn, validation_split=0.0, flip=False, shift = shift, onelabel=onelabel, seed=seed)# ,single_img=single_im

for i in range(int(126*1.5)):
    imgs, masks, paths = train.__getrawitem__(i)
    for j in range(16):
        if np.all(masks[j]==0):
            print(paths[j][0])

# from tensorflow.keras.preprocessing.image import load_img
# import numpy as np
# image = load_img("labels/ckon8q10y0004266duu4ir7t2.png", color_mode="grayscale")
# "labels/ckoka7iqa00003g68drm4onrw.png"
# image = np.where(np.array(image) == 0, 0, 1)
# plt.imshow(image)
# plt.show()
