import tensorflow as tf
import numpy as np


#DATA GENERATOR
img_dir = "images/"
mask_dir = "labels/"
image_extension = ".png"
mask_extension = ".png"
batch_size = 16
horizontal_split = 12 #1
vertical_split = 1

seed = 42
onelabel = True
shift = True
single_img = False

#PREPROCESS FUNCTION OF THE PRETRAINED ENCODER
#from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess
#from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess
preprocess_fcn = preprocess



# TRAINING 
def jaccard_distance(y_true, y_pred, smooth=100):

    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    
    """ Calculates mean of Jaccard distance as a loss function """
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
    sum_ = tf.reduce_sum(y_true + y_pred, axis=(1,2))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    jd =  (1 - jac) * smooth
    return tf.reduce_mean(jd)

# import tensorflow.keras.backend as K
# def IoULoss(targets, inputs, smooth=1e-6):
    
#     #flatten label and prediction tensors
#     inputs = K.flatten(inputs)
#     targets = K.flatten(targets)
    
#     intersection = K.sum(K.dot(targets, inputs))
#     total = K.sum(targets) + K.sum(inputs)
#     union = total - intersection
    
#     IoU = (intersection + smooth) / (union + smooth)
#     return 1 - IoU


import tensorflow.keras.losses as losses
#loss = losses.SparseCategoricalCrossentropy()
loss = losses.BinaryCrossentropy()
#loss = jaccard_distance
#loss = IoULoss()



epochs=50
steps_per_epoch=20
#callback = None
from tensorflow.keras.callbacks import EarlyStopping
callback = EarlyStopping(monitor="loss",
     min_delta=0.01,
     patience=5,
     verbose=1,
     mode="auto",
     baseline=None,
     restore_best_weights=False)



#RESULTS
dir_name = "results/samuel/ResNet50V2_wS_SC_ES_50epochs"
