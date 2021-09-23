# Structure + Layers:
# https://github.com/Runist/SegNet-keras/blob/master/nets/SegNet.py

from segnetLayers import MaxPoolingWithIndices2D, MaxUnpoolingWithIndices2D
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def SegNet(input_shape, num_classes):
    """
    论文中介绍的SegNet网络
    :param input_shape: 模型输入shape
    :param num_classes: 分类数量
    :return: model
    """
    inputs = layers.Input(shape=input_shape)

    # encoder
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x, mask_1 = MaxPoolingWithIndices2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x, mask_2 = MaxPoolingWithIndices2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x, mask_3 = MaxPoolingWithIndices2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x, mask_4 = MaxPoolingWithIndices2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x, mask_5 = MaxPoolingWithIndices2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.5)(x)

    # decoder
    x = MaxUnpoolingWithIndices2D((2, 2))([x, mask_5])
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    x = MaxUnpoolingWithIndices2D((2, 2))([x, mask_4])
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    x = MaxUnpoolingWithIndices2D((2, 2))([x, mask_3])
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    x = MaxUnpoolingWithIndices2D((2, 2))([x, mask_2])
    x = layers.Convolution2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Convolution2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = MaxUnpoolingWithIndices2D((2, 2))([x, mask_1])
    x = layers.Convolution2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_classes, (1, 1), padding='valid', kernel_initializer='he_uniform')(x)
    outputs = layers.BatchNormalization()(x)

    outputs = layers.Activation('softmax')(x)

    segnet_model = Model(inputs=inputs, outputs=outputs, name='SegNet')

    return segnet_model