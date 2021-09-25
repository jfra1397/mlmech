import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K


#DATA GENERATOR
img_dir = "images/"
mask_dir = "labels/"
image_extension = ".png"
mask_extension = ".png"
img_size = (256, 256, 3)
batch_size = 16
horizontal_split = 12
vertical_split = 1
val_split = 0.1
seed = 42
onelabel = False
shift = False
single_img = False

#PREPROCESS FUNCTION OF THE PRETRAINED ENCODER
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess
#from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess
#from tensorflow.python.keras.applications.vgg16 import preprocess_input as preprocess
preprocess_fcn = preprocess



# TRAINING 
def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    # y_true = tf.keras.layers.Flatten()(y_true)
    # y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def dice_metric(y_pred, y_true):
    intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
    union = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))
    # if y_pred.sum() == 0 and y_pred.sum() == 0:
    #     return 1.0

    return 2*intersection / union

import tensorflow.keras.losses as losses
#loss = losses.SparseCategoricalCrossentropy()
#loss = losses.BinaryCrossentropy()
#loss = jaccard_distance
loss = jaccard_distance_loss



epochs=30
steps_per_epoch=20
callback = None
#from tensorflow.keras.callbacks import EarlyStopping
# callback = EarlyStopping(monitor="loss",
#      min_delta=0.01,
#      patience=5,
#      verbose=1,
#      mode="auto",
#      baseline=None,
#      restore_best_weights=False)



#RESULTS
dir_name = "results/julian/unet_jaccard"
