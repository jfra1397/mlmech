
#DATA GENERATOR
seed = 42
onelabel = False
shift = 1
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess
#from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess
#from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess
preprocess_fcn = preprocess



# TRAINING
import tensorflow.keras.losses as losses
loss = losses.SparseCategoricalCrossentropy()
#loss = losses.BinaryCrossentropy()
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
# loss = IoULoss()

epochs=50
steps_per_epoch=20
from tensorflow.keras.callbacks import EarlyStopping
callback = None#EarlyStopping(monitor="loss",
    # min_delta=0.01,
    # patience=5,
    # verbose=1,
    # mode="auto",
    # baseline=None,
    # restore_best_weights=False)



#RESULTS
dir_name = "results/julian/vgg16_5"