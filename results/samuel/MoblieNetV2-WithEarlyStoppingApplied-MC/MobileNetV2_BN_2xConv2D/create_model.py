import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Concatenate
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose
from tensorflow.python.keras.backend import conv3d_transpose

def generate_model(img_size):
    
    encoder = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(img_size), classifier_activation=None)
    # encoder = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_tensor=None, input_shape=(img_size), pooling=None, classifier_activation= None)
    # encoder = tf.keras.applications.ResNet50V5(include_top=False, weights="imagenet", input_tensor=None, input_shape=(img_size), pooling=None, classifier_activation= None)
    encoder.trainable = False

    # d1 = UpSampling2D(size=(2, 2))(encoder.layers[-1].output)
    # c1 = Conv2D(8, kernel_size=(3, 3), activation='selu', padding='same')(d1)
    # d2 = UpSampling2D(size=(2, 2))(c1)
    # c2 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d2)
    # d3 = UpSampling2D(size=(2, 2))(c2)
    # c3 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d3)
    # d4 = UpSampling2D(size=(2, 2))(c3)
    # c4 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d4)
    # d5 = UpSampling2D(size=(2, 2))(c4)
    # #c5 = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same')(d5)
    # c5 = Conv2D(3, kernel_size=(1, 1), activation='softmax', padding='same')(d5)
    # output = c5

#BatchNormalization
    # d1 = UpSampling2D(size=(2, 2))(encoder.layers[-1].output)
    # c1 = Conv2D(8, kernel_size=(3, 3), activation='selu', padding='same')(d1)
    # b1 = BatchNormalization()(c1)
    # d2 = UpSampling2D(size=(2, 2))(b1)
    # c2 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d2)
    # b2 = BatchNormalization()(c2)
    # d3 = UpSampling2D(size=(2, 2))(b2)
    # c3 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d3)
    # b3 = BatchNormalization()(c3)
    # d4 = UpSampling2D(size=(2, 2))(b3)
    # c4 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d4)
    # b4 = BatchNormalization()(c4)
    # d5 = UpSampling2D(size=(2, 2))(b4)
    # #c5 = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same')(d5)
    # c5 = Conv2D(3, kernel_size=(1, 1), activation='softmax', padding='same')(d5)
    # output = c5


    # model = Model(inputs=encoder.inputs, outputs=output)

#BatchNormalization, Conv2DTranspose
    # d1 = Conv2DTranspose(16, (3, 3), strides=2, activation="relu", padding="same")(encoder.layers[-1].output)
    # c1 = Conv2D(8, kernel_size=(3, 3), activation='selu', padding='same')(d1)
    # b1 = BatchNormalization()(c1)
    # d2 = Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(b1)
    # c2 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d2)
    # b2 = BatchNormalization()(c2)
    # d3 = Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(b2)
    # c3 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d3)
    # b3 = BatchNormalization()(c3)
    # d4 = Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(b3)
    # c4 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d4)
    # b4 = BatchNormalization()(c4)
    # d5 = Conv2DTranspose(256, (3, 3), strides=2, activation="relu", padding="same")(b4)
    # #c5 = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same')(d5)
    # c5 = Conv2D(3, kernel_size=(1, 1), activation='softmax', padding='same')(d5)
    # output = c5


    # model = Model(inputs=encoder.inputs, outputs=output)

#BatchNormalization, 2xConv2Layers

    d1 = UpSampling2D(size=(2, 2))(encoder.layers[-1].output)
    c11 = Conv2D(8, kernel_size=(3, 3), activation='selu', padding='same')(d1)
    b11 = BatchNormalization()(c11)
    c12 = Conv2D(8, kernel_size=(3, 3), activation='selu', padding='same')(b11)
    b12 = BatchNormalization()(c12)

    d2 = UpSampling2D(size=(2, 2))(b12)
    c21 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d2)
    b21 = BatchNormalization()(c21)
    c22 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(b21)
    b22 = BatchNormalization()(c22)

    d3 = UpSampling2D(size=(2, 2))(b22)
    c31 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d3)
    b31 = BatchNormalization()(c31)
    c32 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(b31)
    b32 = BatchNormalization()(c32)

    d4 = UpSampling2D(size=(2, 2))(b32)
    c41 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d4)
    b41 = BatchNormalization()(c41)
    c42 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(b41)
    b42 = BatchNormalization()(c42)

    d5 = UpSampling2D(size=(2, 2))(b42)
    #c5 = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same')(d5)
    c5 = Conv2D(3, kernel_size=(1, 1), activation='softmax', padding='same')(d5)
    output = c5


    model = Model(inputs=encoder.inputs, outputs=output)

#Conv2DTranspose
    # d1 = Conv2DTranspose(16, (3, 3), strides=2, activation="relu", padding="same")(encoder.layers[-1].output)
    # c1 = Conv2D(8, kernel_size=(3, 3), activation='selu', padding='same')(d1)
    # d2 = Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(c1)
    # c2 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d2)
    # d3 = Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(c2)
    # c3 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d3)
    # d4 = Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(c3)
    # c4 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d4)
    # d5 = Conv2DTranspose(256, (3, 3), strides=2, activation="relu", padding="same")(c4)
    # #c5 = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same')(d5)
    # c5 = Conv2D(3, kernel_size=(1, 1), activation='softmax', padding='same')(d5)
    # output = c5


    # model = Model(inputs=encoder.inputs, outputs=output)

#2xConv2Layers
    # d1 = UpSampling2D(size=(2, 2))(encoder.layers[-1].output)
    # c11 = Conv2D(8, kernel_size=(3, 3), activation='selu', padding='same')(d1)
    # c12 = Conv2D(8, kernel_size=(3, 3), activation='selu', padding='same')(c11)

    # d2 = UpSampling2D(size=(2, 2))(c12)
    # c21 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d2)
    # c22 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(c21)

    # d3 = UpSampling2D(size=(2, 2))(c22)
    # c31 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d3)
    # c32 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(c31)

    # d4 = UpSampling2D(size=(2, 2))(c32)
    # c41 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d4)
    # c42 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(c41)

    # d5 = UpSampling2D(size=(2, 2))(c42)
    # #c5 = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same')(d5)
    # c5 = Conv2D(3, kernel_size=(1, 1), activation='softmax', padding='same')(d5)
    # output = c5


    # model = Model(inputs=encoder.inputs, outputs=output)

#Unet Architecture
    
    # complexity = 5

    # x = Input((img_size))
    # inputs = x

    # #down sampling 
    # f = 8
    # layers = []

    # for i in range(0, complexity):
    #     x = Conv2D(f, 3, activation='relu', padding='same') (x)
    #     x = Conv2D(f, 3, activation='relu', padding='same') (x)
    #     layers.append(x)
    #     x = MaxPooling2D() (x)
    #     f = f*2
    # ff2 = 64 

    # #bottleneck 
    # j = len(layers) - 1
    # x = Conv2D(f, 3, activation='relu', padding='same') (x)
    # x = Conv2D(f, 3, activation='relu', padding='same') (x)
    # x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same') (x)
    # x = Concatenate(axis=3)([x, layers[j]])
    # j = j -1 

    # #upsampling 
    # for i in range(0, complexity-1):
    #     ff2 = ff2//2
    #     f = f // 2 
    #     x = Conv2D(f, 3, activation='relu', padding='same') (x)
    #     x = Conv2D(f, 3, activation='relu', padding='same') (x)
    #     x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same') (x)
    #     x = Concatenate(axis=3)([x, layers[j]])
    #     j = j -1 

    # #classification 
    # outputs = Conv2D(1, 1, activation='sigmoid') (x)

    # #model creation 
    # model = Model(inputs=[inputs], outputs=[outputs])


    #BatchNormilzedlayer
    #GeneralAveragePooling
    #Dropout
    #NoShift

    return model