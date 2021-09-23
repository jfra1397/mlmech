from enum import Flag
from os import listdir
from os.path import isfile, join
import random
from traceback import print_tb

import matplotlib.pyplot as plt

from tensorflow.keras.utils import Sequence

import numpy as np
from tensorflow.keras.preprocessing.image import load_img




class CustomDataGeneratorPredict(Sequence):

    def __init__(self, data_list, img_dir, mask_dir, horizontal_split, vertical_split,img_extension, mask_extension,process_fcn, onelabel) -> None:
        super().__init__()
        self.batch_size = len(data_list)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_extension = img_extension
        self.mask_extension = mask_extension
        self.horizontal_split = horizontal_split
        self.vertical_split = vertical_split
        self.classes = np.array([])
        self.process = process_fcn
        self.onelabel = onelabel
        self.data = data_list
        print("Number of samples: ", len(self.data))
        if self.onelabel:
            self.classes = np.array([0,1])
        else:
            self._count_classes(data_list)

        print(f"Classes: {self.classes}")
        self.__getrawitem__()

    def _count_classes(self, files):
        for file in files:
            mask = load_img(self.mask_dir + file[0] + self.mask_extension, color_mode="grayscale")
            mask_array = np.array(mask)
            classes = np.unique(mask_array.flatten())
            for value in classes:
              if not np.any(self.classes==value):
                self.classes = np.append(self.classes, value)
                self.classes = np.sort(self.classes)

    def __getrawitem__(self):
        """Returns tuple (input, target) correspond to batch #idx."""
        x = []
        y = []
        for data in self.data:
            image = load_img(self.img_dir + data[0] + self.img_extension)
            mask = load_img(self.mask_dir + data[0] + self.mask_extension, color_mode="grayscale")
            image_array = np.array(image)
            mask_array = np.array(mask)
            if self.onelabel:
                mask_array = np.where(mask_array == 0, 0, 1)
            else:
                for i in range(len(self.classes)):
                    mask_array = np.where(mask_array==self.classes[i], i, mask_array)
            image_array_split = np.split(np.split(image_array, self.vertical_split, axis=0)[data[2]], self.horizontal_split, axis=1)[data[1]]
            mask_array_split = np.split(np.split(mask_array, self.vertical_split, axis=0)[data[2]], self.horizontal_split, axis=1)[data[1]]

            self.img_size = image_array_split.shape
            self.mask_size = mask_array_split.shape

            x.append(image_array_split)
            y.append(mask_array_split)

        return x,y

    def __getitem__(self):

        imgs, masks = self.__getrawitem__()

        x = []
        for img in imgs:
            x.append(self.process(img))
            #x.append(((img/255) - 0.5) * 2)
        y = []
        for mask in masks:
            y.append(mask)
        
        # if not self.single_img and not self.onelabel: 
        #     y = np.array(y)
        #     y_temp = np.zeros((*y.shape,3), dtype=np.uint16)
        #     y_temp[:,:,:,0] = np.where(y == 0,1,0)
        #     y_temp[:,:,:,1] = np.where(y == 1,1,0)
        #     y_temp[:,:,:,2] = np.where(y == 2,1,0)

        #     y = y_temp
        
        return np.array(x), np.array(y, dtype=np.float32)

    def plot_batch(self):
        img, mask = self.__getrawitem__()
        fig,axs = plt.subplots(self.batch_size,2, figsize=(10, self.batch_size*5))
        axs = axs.flatten()

        for i in range(self.batch_size):
            axs[2*i].imshow(img[i])
            axs[2*i+1].imshow(mask[i], vmin=0, vmax = len(self.classes))
            
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    def plot_prediction(self, predictions):
        img, mask = self.__getrawitem__()
        fig,axs = plt.subplots(self.batch_size,3, figsize=(15, self.batch_size*5))
        axs[0][0].set_title('Real Image')
        axs[0][1].set_title('Mask')
        axs[0][2].set_title('Predictions')
        axs = axs.flatten()

        for i in range(self.batch_size):
            axs[3*i].imshow(img[i])
            axs[3*i+1].imshow(mask[i], vmin=0, vmax = len(self.classes))
            if predictions[i].shape[2] > 1:
                class_0 = predictions[i][:,:,0]
                class_1 = predictions[i][:,:,1]
                class_2 = predictions[i][:,:,2]
                result = 0*(np.where(class_1>class_2,class_1,class_2) < class_0) +  1*(np.where(class_0>class_2,class_0,class_2) < class_1) +  2*(np.where(class_1>class_0,class_1,class_0) < class_2)
            else:
                result = predictions[i].reshape(self.mask_size)
            
            axs[3*i+2].imshow(result, vmin=0, vmax = len(self.classes))
            
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()

    def save_batch(self, path, net_name = None, predictions = None):
        img, mask = self.__getrawitem__()
        for i in range(self.batch_size):
            with open(path + f"/img_{i}.npy", "wb") as f:
                np.save(f, np.array(img[i]))
            with open(path + f"/mask_{i}.npy", "wb") as f:
                np.save(f, np.array(mask[i]))
            if predictions is not None:
                if predictions[i].shape[2] > 1:
                    class_0 = predictions[i][:,:,0]
                    class_1 = predictions[i][:,:,1]
                    class_2 = predictions[i][:,:,2]
                    result = 0*(np.where(class_1>class_2,class_1,class_2) < class_0) +  1*(np.where(class_0>class_2,class_0,class_2) < class_1) +  2*(np.where(class_1>class_0,class_1,class_0) < class_2)
                else:
                    result = predictions[i].reshape(self.mask_size)
                with open(path + f"/{net_name}_pred_{i}.npy", "wb") as f:
                    np.save(f, np.array(result))





if __name__ == "__main__":
    
    img_dir = "PredImages/"
    mask_dir = "PredLabels/"
    image_extension = ".png"
    mask_extension = ".png"
    horizontal_split = 12
    vertical_split = 1
    onelabel = False
    #from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess
    from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess
    #from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess
    preprocess_fcn = preprocess
    data_list = [["ckokey6vd002b3g68q5mmclyi", 3, 0],
                    ["ckoklbzlb06qe3g688yq4w56s", 7, 0],
                    ["ckokf3nzs00473g68e456ksi3", 1, 0]]

    test = CustomDataGeneratorPredict(data_list, img_dir, mask_dir, horizontal_split, vertical_split, image_extension, mask_extension, preprocess_fcn, onelabel)

    test.plot_batch()

    img, mask = test.__getitem__()
    print(img.min())
    print(img.max())
    print(mask.min())
    print(mask.max())
    print(test.classes)




