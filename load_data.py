from enum import Flag
from os import listdir
from os.path import isfile, join
import random

import matplotlib.pyplot as plt

from tensorflow.keras.utils import Sequence

import numpy as np
from tensorflow.keras.preprocessing.image import load_img


class CustomDataGenerator(Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    @classmethod
    def generate_data(cls, batch_size, img_size, img_dir, mask_dir, horizontal_split, 
                    vertical_split, img_extension, mask_extension, process_fcn, validation_split = 0.2, flip = True, shift = 0, onelabel = False, seed=None):
        onlyfiles_images = [f for f in listdir(img_dir) if isfile(join(img_dir, f)) if f.endswith(img_extension)]
        onlyfiles_masks = [f for f in listdir(mask_dir) if isfile(join(mask_dir, f)) if f.endswith(mask_extension)]
        onlyfiles = [a[:-4] for a in onlyfiles_masks for b in onlyfiles_images if a[:-4] == b[:-4]]
        split = int(len(onlyfiles)*(1-validation_split))
        test_onlyfiles = onlyfiles[:split]
        val_onlyfiles = onlyfiles[split:]
        test = cls(batch_size, img_size, img_dir, mask_dir, horizontal_split, vertical_split, img_extension, mask_extension, test_onlyfiles, process_fcn, flip = flip, shift = shift, onelabel = onelabel, seed=seed)
        val = cls(batch_size, img_size, img_dir, mask_dir, horizontal_split, vertical_split, img_extension, mask_extension, val_onlyfiles, process_fcn, flip = flip, shift = shift, onelabel = onelabel, seed = seed)

        return test,val
    
    def __init__(self, batch_size, img_size, img_dir, mask_dir, horizontal_split, vertical_split, img_extension, mask_extension, onlyfiles, process_fcn, flip = True, shift = 0, onelabel = False, seed=None):
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_extension = img_extension
        self.mask_extension = mask_extension
        self.horizontal_split = horizontal_split
        self.vertical_split = vertical_split
        self.classes = np.array([])
        self.process = process_fcn
        self.onelabel = onelabel

        hshift = 4
        vshift = 4

        flip = [True, False] if flip else [False]
        if shift > 1: 
            shift = 1
        if shift <= 0:
            shift = 0
            hshift = 1
            vshift = 1


        self.data = []
        for files in onlyfiles:
            for i in range(horizontal_split):
                for j in range(vertical_split):
                    for hflip in flip:
                        for vflip in flip:
                            for _ in range(hshift):
                                for _ in range(vshift):
                                    self.data.append([files, i, j, hflip, vflip, random.uniform(0,shift), random.uniform(0,shift)])

        random.seed(seed)
        random.shuffle(self.data)
        print("Number of samples: ", len(self.data))
        if self.onelabel:
            self.classes = np.array([0,1])
        else:
            self._count_classes(onlyfiles)

        print(f"Classes: {self.classes}")
    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, idx):

        imgs, masks = self.__getrawitem__(idx)

        x = []
        for img in imgs:
            x.append(self.process(img))
            #x.append(((img/255) - 0.5) * 2)
        y = []
        for mask in masks:
            y.append(mask)
        
        return np.array(x), np.array(y)

    def get_path(self, idx):
        i = idx * self.batch_size
        batch_paths = self.data[i : i + self.batch_size]
        for data in batch_paths:
            print(data)

    def __getrawitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_paths = self.data[i : i + self.batch_size]
        x = []
        y = []
        for data in batch_paths:
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
            image_array_split = np.flip(image_array_split, axis=0) if data[3] else image_array_split
            image_array_split = np.flip(image_array_split, axis=1) if data[4] else image_array_split
            mask_array_split = np.flip(mask_array_split, axis=0) if data[3] else mask_array_split
            mask_array_split = np.flip(mask_array_split, axis=1) if data[4] else mask_array_split

            image_array_split = np.roll(image_array_split, int(image_array_split.shape[0]*data[5]), axis=0)
            image_array_split = np.roll(image_array_split, int(image_array_split.shape[1]*data[6]), axis=1)
            mask_array_split = np.roll(mask_array_split, int(mask_array_split.shape[0]*data[5]), axis=0)
            mask_array_split = np.roll(mask_array_split, int(mask_array_split.shape[1]*data[6]), axis=1)

            
            x.append(image_array_split)
            y.append(mask_array_split)
        
        return x,y

    def _count_classes(self, files):
        for file in files:
            mask = load_img(self.mask_dir + file + self.mask_extension, color_mode="grayscale")
            mask_array = np.array(mask)
            classes = np.unique(mask_array.flatten())
            for value in classes:
              if not np.any(self.classes==value):
                self.classes = np.append(self.classes, value)
                self.classes = np.sort(self.classes)


    
    def plot_batch(self, idx):
        img, mask = self.__getrawitem__(idx)
        fig,axs = plt.subplots(self.batch_size,2, figsize=(10, self.batch_size*5))
        axs = axs.flatten()

        for i in range(self.batch_size):
            axs[2*i].imshow(img[i])
            axs[2*i+1].imshow(mask[i], vmin=0, vmax = len(self.classes))
            
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()

    def plot_prediction(self, idx, predictions):
        img, mask = self.__getrawitem__(idx)
        fig,axs = plt.subplots(self.batch_size,3, figsize=(15, self.batch_size*5))
        axs = axs.flatten()

        for i in range(self.batch_size):
            axs[3*i].imshow(img[i])
            axs[3*i+1].imshow(mask[i], vmin=0, vmax = len(self.classes))
            axs[3*i+2].imshow(predictions[i].reshape(256,256), vmin=0, vmax = len(self.classes))
            
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()