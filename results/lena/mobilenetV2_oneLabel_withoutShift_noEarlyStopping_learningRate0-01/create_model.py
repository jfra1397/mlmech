import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten

def generate_model():
    
    encoder = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(256, 256, 3), classifier_activation=None)
    encoder.trainable = False

    d1 = UpSampling2D(size=(2, 2))(encoder.layers[-1].output)
    c1 = Conv2D(8, kernel_size=(3, 3), activation='selu', padding='same')(d1)
    d2 = UpSampling2D(size=(2, 2))(c1)
    c2 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d2)
    d3 = UpSampling2D(size=(2, 2))(c2)
    c3 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d3)
    d4 = UpSampling2D(size=(2, 2))(c3)
    c4 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d4)
    d5 = UpSampling2D(size=(2, 2))(c4)
    c5 = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same')(d5)
    #c5 = Conv2D(3, kernel_size=(1, 1), activation='softmax', padding='same')(d5)
    output = c5

    model = Model(inputs=encoder.inputs, outputs=output)

    

    return model