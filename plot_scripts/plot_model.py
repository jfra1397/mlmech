
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Concatenate
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, add, Activation, Dropout
import visualkeras
from PIL import ImageFont
font = ImageFont.truetype("arial.ttf", 24*2)  # using comic sans is strictly prohibited!


model_type = "simple_decoder" # "vgg", mobilenetv2", "resnet", "simple_decoder", "unet", "advanced_decoder"

if model_type == "vgg":
    encoder = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_tensor=None, input_shape=(256,256,3), pooling=None, classifier_activation= None)
    model = Model(inputs=encoder.inputs, outputs=encoder.output)
    font = ImageFont.truetype("arial.ttf", 24)  # using comic sans is strictly prohibited!
    visualkeras.layered_view(model, legend=True, scale_xy = 1, font=font, to_file="plots/models/vgg16.png").show()
elif model_type == "mobilenetv2":
    encoder = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(256,256,3), classifier_activation=None)
    model = Model(inputs=encoder.inputs, outputs=encoder.output)
    font = ImageFont.truetype("arial.ttf", 24*2)  # using comic sans is strictly prohibited!
    visualkeras.layered_view(model, legend=True, font=font, scale_xy=3, to_file="plots/models/mobilenet_v2.png", scale_z=0.0001, max_z=0.1).show()
elif model_type == "resnet":
    encoder = tf.keras.applications.ResNet50V2(include_top=False, weights="imagenet", input_tensor=None, input_shape=(256,256,3), pooling=None, classifier_activation= None)
    model = Model(inputs=encoder.inputs, outputs=encoder.output)
    font = ImageFont.truetype("arial.ttf", 24*2)  # using comic sans is strictly prohibited
    visualkeras.layered_view(model, legend=True, font=font, scale_xy=3, to_file="plots/models/resnet.png", scale_z=0.0001, max_z=0.1).show()
elif model_type == "simple_decoder":
    x = Input((256,256,3))
    d1 = UpSampling2D(size=(2, 2))(x)
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
    model = Model(inputs=x, outputs=output)
    font = ImageFont.truetype("arial.ttf", 24*3)  # using comic sans is strictly prohibited!
    visualkeras.layered_view(model, legend=True, font=font, scale_xy = 0.1, to_file="plots/models/simple_decoder.png", scale_z = 10, index_ignore=[0]).show()
elif model_type == "unet":
    model = tf.keras.models.load_model("results/julian/unet_4/model.tf")
    font = ImageFont.truetype("arial.ttf", 24)  # using comic sans is strictly prohibited!
    visualkeras.layered_view(model, legend=True, scale_xy = 1, font=font, to_file="plots/models/unet.png").show()
elif model_type == "advanced_decoder":
    x = Input((256,256,3))
    base = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(x)
    t1 = Conv2DTranspose(16, (3, 3), activation="selu", padding="same")(base)
    act11 = Activation("selu")(t1)
    b12 = BatchNormalization()(act11)
    dUp1 = UpSampling2D(size=(2, 2))(b12)
    d1 = UpSampling2D(size=(2, 2))(b12)
    c1 = Conv2D(16, kernel_size=(3, 3), activation='selu', padding='same')(d1)
    plus1 = add([c1,dUp1])
    act12 = Activation("selu")(plus1)
    b12 = BatchNormalization()(act12)
    drop1 = Dropout(0.3)(b12)
    #Layer 2, 3 and 4 are built similar
    #....
    #Output layer, containing a layer for Multi or Single Class
    d5 = Conv2DTranspose(256, (3, 3), strides=2, activation="relu", padding="same")(drop1)                      
    c5 = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same')(d5)
    model = Model(inputs=x, outputs=c5)
    font = ImageFont.truetype("arial.ttf", 24)  # using comic sans is strictly prohibited!
    visualkeras.layered_view(model, legend=True, font=font, scale_xy = 0.1, to_file="plots/models/simple_decoder.png", scale_z = 10, index_ignore=[0]).show()
