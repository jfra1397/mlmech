import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Concatenate
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, add, Activation, Dropout
#from tensorflow.keras.backend import conv3d_transpose

def generate_model(img_size): #,onelabel):
    
    # encoder = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(img_size), classifier_activation=None)
    # #encoder = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_tensor=None, input_shape=(img_size), pooling=None, classifier_activation= None)
    # #encoder = tf.keras.applications.ResNet50V2(include_top=False, weights="imagenet", input_tensor=None, input_shape=(img_size), pooling=None, classifier_activation= None)
    # encoder.trainable = False

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

    # model = Model(inputs=encoder.inputs, outputs=output)
##################### BN
# BatchNormalization:

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
###################################### BN_Conv2DTranspose
# BatchNormalization, Conv2DTranspose:

    # d1 = Conv2DTranspose(16, (3, 3), strides=2, activation="relu", padding="same")(encoder.layers[-1].output)
    # c1 = Conv2D(8, kernel_size=(3, 3), activation='selu', padding='same')(d1)
    # b1 = BatchNormalization()(c1)
    # d2 = Conv2DTranspose(16, (3, 3), strides=2, activation="relu", padding="same")(b1)
    # c2 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d2)
    # b2 = BatchNormalization()(c2)
    # d3 = Conv2DTranspose(16, (3, 3), strides=2, activation="relu", padding="same")(b2)
    # c3 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d3)
    # b3 = BatchNormalization()(c3)
    # d4 = Conv2DTranspose(16, (3, 3), strides=2, activation="relu", padding="same")(b3)
    # c4 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d4)
    # b4 = BatchNormalization()(c4)
    # d5 = Conv2DTranspose(16, (3, 3), strides=2, activation="relu", padding="same")(b4)
    # c5 = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same')(d5)
    # #c5 = Conv2D(3, kernel_size=(1, 1), activation='softmax', padding='same')(d5)
    # output = c5


    # model = Model(inputs=encoder.inputs, outputs=output)
#################################### BN_2xConv2D
# BatchNormalization, 2xConv2Layers:

    # d1 = UpSampling2D(size=(2, 2))(encoder.layers[-1].output)
    # c11 = Conv2D(8, kernel_size=(3, 3), activation='selu', padding='same')(d1)
    # b11 = BatchNormalization()(c11)
    # c12 = Conv2D(8, kernel_size=(3, 3), activation='selu', padding='same')(b11)
    # b12 = BatchNormalization()(c12)

    # d2 = UpSampling2D(size=(2, 2))(b12)
    # c21 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d2)
    # b21 = BatchNormalization()(c21)
    # c22 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(b21)
    # b22 = BatchNormalization()(c22)

    # d3 = UpSampling2D(size=(2, 2))(b22)
    # c31 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d3)
    # b31 = BatchNormalization()(c31)
    # c32 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(b31)
    # b32 = BatchNormalization()(c32)

    # d4 = UpSampling2D(size=(2, 2))(b32)
    # c41 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d4)
    # b41 = BatchNormalization()(c41)
    # c42 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(b41)
    # b42 = BatchNormalization()(c42)

    # d5 = UpSampling2D(size=(2, 2))(b42)
    # #c5 = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same')(d5)
    # c5 = Conv2D(3, kernel_size=(1, 1), activation='softmax', padding='same')(d5)
    # output = c5


    # model = Model(inputs=encoder.inputs, outputs=output)
################## Conv2DTranspose
# Conv2DTranspose:

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
################ 2xConv2D
# 2xConv2Layers:

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
############################# AddUpsampling
# Conv2D, 2x Upsampling, Add:

    # base = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(encoder.layers[-1].output)
    # dUp1 = UpSampling2D(size=(2, 2))(base)
    # d1 = UpSampling2D(size=(2, 2))(base)
    # c1 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d1)
    # plus1 = add([c1,dUp1])
    # act1 = Activation("selu")(plus1)
    # b1 = BatchNormalization()(act1)

    # dUp2 = UpSampling2D(size=(2, 2))(b1)
    # d2 = UpSampling2D(size=(2, 2))(b1)
    # c2 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d2)
    # plus2 = add([c2,dUp2])
    # act2 = Activation("selu")(plus2)
    # b2 = BatchNormalization()(act2)

    # dUp3 = UpSampling2D(size=(2, 2))(b2)
    # d3 = UpSampling2D(size=(2, 2))(b2)
    # c3 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d3)
    # plus3 = add([c3,dUp3])
    # act3 = Activation("selu")(plus3)
    # b3 = BatchNormalization()(act3)

    # dUp4 = UpSampling2D(size=(2, 2))(b3)
    # d4 = UpSampling2D(size=(2, 2))(b3)
    # c4 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d4)
    # plus4 = add([c4,dUp4])
    # act4 = Activation("selu")(plus4)
    # b4 = BatchNormalization()(act4)
    
    # d5 = UpSampling2D(size=(2, 2))(b4)
    # c5 = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same')(d5)
    # #c5 = Conv2D(3, kernel_size=(1, 1), activation='softmax', padding='same')(d5)
    # # output = c5


    # model = Model(inputs=encoder.inputs, outputs=output)

########################################### AddTranspose
# Conv2DTranspose, Conv2D, Upsampling, Add:

    # base = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(encoder.layers[-1].output)
    # dUp1 = UpSampling2D(size=(2, 2))(base)
    # d1 = Conv2DTranspose(16, (3, 3), strides=2, activation="relu", padding="same")(base)
    # c1 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d1)
    # plus1 = add([c1,dUp1])
    # act1 = Activation("selu")(plus1)
    # b1 = BatchNormalization()(act1)

    # dUp2 = UpSampling2D(size=(2, 2))(b1)
    # d2 = Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(b1)
    # c2 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d2)
    # plus2 = add([c2,dUp2])
    # act2 = Activation("selu")(plus2)
    # b2 = BatchNormalization()(act2)

    # dUp3 = UpSampling2D(size=(2, 2))(b2)
    # d3 = Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(b2)
    # c3 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d3)
    # plus3 = add([c3,dUp3])
    # act3 = Activation("selu")(plus3)
    # b3 = BatchNormalization()(act3)

    # dUp4 = UpSampling2D(size=(2, 2))(b3)
    # d4 = Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(b3)
    # c4 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d4)
    # plus4 = add([c4,dUp4])
    # act4 = Activation("selu")(plus4)
    # b4 = BatchNormalization()(act4)
    
    # d5 = Conv2DTranspose(256, (3, 3), strides=2, activation="relu", padding="same")(b4)
    # c5 = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same')(d5)
    # #c5 = Conv2D(3, kernel_size=(1, 1), activation='softmax', padding='same')(d5)
    # output = c5


    # model = Model(inputs=encoder.inputs, outputs=output)

################################################### AddDropout
#Conv2DTranspose, Conv2D, Upsampling, Add, Dropout:

    # base = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(encoder.layers[-1].output)
    # dUp1 = UpSampling2D(size=(2, 2))(base)
    # d1 = Conv2DTranspose(16, (3, 3), strides=2, activation="selu", padding="same")(base)
    # c1 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d1)
    # plus1 = add([c1,dUp1])
    # act1 = Activation("selu")(plus1)
    # b1 = BatchNormalization()(act1)
    # drop1 = Dropout(0.3)(b1)

    # dUp2 = UpSampling2D(size=(2, 2))(drop1)
    # d2 = Conv2DTranspose(32, (3, 3), strides=2, activation="selu", padding="same")(drop1)
    # c2 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d2)
    # plus2 = add([c2,dUp2])
    # act2 = Activation("selu")(plus2)
    # b2 = BatchNormalization()(act2)
    # drop2 = Dropout(0.3)(b2)
    
    # dUp3 = UpSampling2D(size=(2, 2))(drop2)
    # d3 = Conv2DTranspose(64, (3, 3), strides=2, activation="selu", padding="same")(drop2)
    # c3 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d3)
    # plus3 = add([c3,dUp3])
    # act3 = Activation("selu")(plus3)
    # b3 = BatchNormalization()(act3)
    # drop3 = Dropout(0.3)(b3)
    
    # dUp4 = UpSampling2D(size=(2, 2))(drop3)
    # d4 = Conv2DTranspose(128, (3, 3), strides=2, activation="selu", padding="same")(drop3)
    # c4 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d4)
    # plus4 = add([c4,dUp4])
    # act4 = Activation("selu")(plus4)
    # b4 = BatchNormalization()(act4)
    # drop4 = Dropout(0.3)(b4)
    
    # d5 = Conv2DTranspose(256, (3, 3), strides=2, activation="selu", padding="same")(drop4)
    # c5 = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same')(d5)
    # #c5 = Conv2D(3, kernel_size=(1, 1), activation='softmax', padding='same')(d5)
    # output = c5


    # model = Model(inputs=encoder.inputs, outputs=output)

################################## KerasModel
# KERAS-Sample Structure Approach:

    # base = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(encoder.layers[-1].output)
    # t1 = Conv2DTranspose(16, (3, 3), activation="selu", padding="same")(base)
    # b12 = BatchNormalization()(t1)
    # act11 = Activation("selu")(b12)
    # dUp1 = UpSampling2D(size=(2, 2))(act11)
    # d1 = UpSampling2D(size=(2, 2))(act11)
    # c1 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d1)
    # plus1 = add([c1,dUp1])
    # b12 = BatchNormalization()(plus1)
    # act12 = Activation("selu")(b12)
    # drop1 = Dropout(0.3)(act12)

    # t2 = Conv2DTranspose(16,(3, 3),activation="selu", padding="same")(drop1)
    # b22 = BatchNormalization()(t2)
    # act21 = Activation("selu")(b22)
    # dUp2 = UpSampling2D(size=(2, 2))(act21)
    # d2 = UpSampling2D(size=(2, 2))(act21)
    # c2 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d2)
    # plus2 = add([c2,dUp2])
    # b22 = BatchNormalization()(plus2)
    # act22 = Activation("selu")(b22)
    # drop2 = Dropout(0.3)(act22)

    # t3 = Conv2DTranspose(16, (3, 3), activation="selu", padding="same")(drop2)
    # b32 = BatchNormalization()(t3)
    # act31 = Activation("selu")(b32)
    # dUp3 = UpSampling2D(size=(2, 2))(act31)
    # d3 = UpSampling2D(size=(2, 2))(act31)
    # c3 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d3)
    # plus3 = add([c3,dUp3])
    # b32 = BatchNormalization()(plus3)
    # act32 = Activation("selu")(b32)
    # drop3 = Dropout(0.3)(act32)
   
    # t4 = Conv2DTranspose(16, (3, 3), activation="selu", padding="same")(drop3)
    # b42 = BatchNormalization()(t4)
    # act41 = Activation("selu")(b42)
    # dUp4 = UpSampling2D(size=(2, 2))(act41)
    # d4 = UpSampling2D(size=(2, 2))(act41)
    # c4 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d4)
    # plus4 = add([c4,dUp4])
    # b42 = BatchNormalization()(plus4)
    # act42 = Activation("selu")(b42)
    # drop4 = Dropout(0.3)(act42)

    # d5 = Conv2DTranspose(256, (3, 3), strides=2, activation="selu", padding="same")(drop4)
    # #c5 = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same')(d5)
    # c5 = Conv2D(3, kernel_size=(1, 1), activation='softmax', padding='same')(d5)
    # output = c5


    # model = Model(inputs=encoder.inputs, outputs=output)
########################################################
################################## KerasModel
# Adanced Decoder Structure Approach:
    encoder = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(img_size), classifier_activation=None)
    encoder.trainable = False
    skip_connection_names = ["input_image", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu", "block_13_expand_relu"]
    # x = encoder.get_layer(skip_connection_names[4]).output
    # f = [32,64,128,256]
    # d = [16,16,32,48]
    x = encoder.layers[-1].output
    f = [16,32,64,128,256]
    d = [16,16,32,48,64]
    ####### Layer i #######
    for i in range(len(f)):
        dUp = UpSampling2D(size=(2, 2))(x)
        dUp = Conv2D(d[-i], kernel_size=(3, 3),activation='selu',padding='same')(dUp)
        dUp = BatchNormalization()(dUp)
        dUp = Activation("selu")(dUp)

        x = Conv2DTranspose(f[i], (3, 3), strides=2, activation="selu", padding="same")(x)
        x = Conv2D(d[-i], kernel_size=(3, 3),activation='selu',padding='same')(x)
        x = Conv2D(d[-i], kernel_size=(3, 3),activation='selu',padding='same')(x)
        x = BatchNormalization()(x)

        x = add([x,dUp])
        x = Activation("selu")(x)
        x = Dropout(0.3)(x)
        
    #c5 = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same')(x)
    c5 = Conv2D(3, kernel_size=(1, 1), activation='softmax', padding='same')(x)
    output =c5
    model = Model(inputs=encoder.inputs, outputs=output)
########################################################
########## U-NET ARCHITECUTRE ##########################
    
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



    ######## UNET + MOBILENET #########
    # inputs = Input(shape=(256, 256, 3), name="input_image")
    # ### Encoder ###
    # encoder = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_tensor=inputs, classifier_activation=None)
    # encoder.trainable = False
    # skip_connection_names = ["input_image", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu", "block_13_expand_relu"]
    # encoder_output = encoder.get_layer("out_relu").output
    # ### Decoder ###
    # f = [16, 32, 48, 64, 64]
    # x = encoder_output
    # for i in range(1, len(skip_connection_names)+1, 1):
    #     x_skip = encoder.get_layer(skip_connection_names[-i]).output
    #     #print(skip_connection_names[-i])
    #     x = UpSampling2D((2, 2))(x)
    #     x = Concatenate()([x, x_skip])
        
    #     x = Conv2D(f[-i], (3, 3), padding="same")(x)
    #     x = BatchNormalization()(x)
    #     x = Activation("relu")(x)
        
    #     x = Conv2D(f[-i], (3, 3), padding="same")(x)
    #     x = BatchNormalization()(x)
    #     x = Activation("relu")(x)
        
    # x = Conv2D(1, (1, 1), padding="same")(x)
    # x = Activation("sigmoid")(x)
    
    # model = Model(encoder.inputs, x)


    return model