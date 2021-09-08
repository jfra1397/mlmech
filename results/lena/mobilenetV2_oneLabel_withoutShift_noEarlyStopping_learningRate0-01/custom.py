
#DATA GENERATOR
seed = 42
onelabel = True
shift = 0#1
#from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess
#from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess
preprocess_fcn = preprocess



# TRAINING
import tensorflow.keras.losses as losses
#loss = losses.SparseCategoricalCrossentropy()
loss = losses.BinaryCrossentropy()
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

epochs=100
steps_per_epoch=20
from tensorflow.keras.callbacks import EarlyStopping
callback = EarlyStopping(monitor="loss",
    min_delta=0.01,
    patience=10,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False)



#RESULTS
dir_name = "results/lena/mobilenetV2_oneLabel_noShift_noEarlyStopping_learningRate0-01"