import tensorflow as tf
from load_data import CustomDataGenerator
import numpy as np

### LENA
#model_path = r'results\lena\mobilenetV2\oneLabel\standard_V2\model.tf'
#savetitle1 = 'results/Plots/Prediction/Preds_MobileNetV2_wS_SC_ES_50epochs-1'
#savetitle2 = 'results/Plots/Prediction/Preds_MobileNetV2_wS_SC_ES_50epochs-2'

#model_path = r'results\lena\mobilenetV2\multiLabel\standard\model.tf'
#savetitle1 = 'results/Plots/Prediction/Preds_MobileNetV2_wS_MC_ES_50epochs-1'
#savetitle2 = 'results/Plots/Prediction/Preds_MobileNetV2_wS_MC_ES_50epochs-2'

#model_path = r'results\lena\mobilenetV2\oneLabel\withoutShift\noEarlyStopping\epochs100_validationSplit0-1\model.tf'
#savetitle1 = 'results/Plots/Prediction/Preds_MobileNetV2_nS_SC_100epochs-1'
#savetitle2 = 'results/Plots/Prediction/Preds_MobileNetV2_nS_SC_100epochs-2'

#model_path = r'results\lena\mobilenetV2\multiLabel\withoutShift\noEarlyStopping_epochs100_validationSplit0-1\model.tf'
#savetitle1 = 'results/Plots/Prediction/Preds_MobileNetV2_nS_MC_100epochs-1'
#savetitle2 = 'results/Plots/Prediction/Preds_MobileNetV2_nS_MC_100epochs-2'


### JULIAN
#model_path = r'results\julian\vgg16_1\model.tf'
#savetitle1 = 'results/Plots/Prediction/Preds_VGG16_wS_SC_ES_50epochs-1'
#savetitle2 = 'results/Plots/Prediction/Preds_VGG16_wS_SC_ES_50epochs-2'

#model_path = r'results\julian\vgg16_3\model.tf'
#savetitle1 = 'results/Plots/Prediction/Preds_VGG16_wS_MC_ES_50epochs-1'
#savetitle2 = 'results/Plots/Prediction/Preds_VGG16_wS_MC_ES_50epochs-2'

#model_path = r'results\julian\vgg16_7\model.tf'
#savetitle1 = 'results/Plots/Prediction/Preds_VGG16_nS_SC_100epochs-1'
#savetitle2 = 'results/Plots/Prediction/Preds_VGG16_nS_SC_100epochs-2'

#model_path = r'results\julian\vgg16_8\model.tf'
#savetitle1 = 'results/Plots/Prediction/Preds_VGG16_nS_MC_100epochs-1'
#savetitle2 = 'results/Plots/Prediction/Preds_VGG16_nS_MC_100epochs-2'


#SAMUEL
# model_path = r'results\samuel\ResNet50V2_wS_SC_ES_50epochs\model.tf'
# savetitle1 = 'results/Plots/Prediction/Preds_ResNet50V2_wS_SC_ES_50epochs-1'
# savetitle2 = 'results/Plots/Prediction/Preds_ResNet50V2_wS_SC_ES_50epochs-2'

# model_path = r'results\samuel\ResNet50V2_wS_MC_ES_50epochs\model.tf'
# savetitle1 = 'results/Plots/Prediction/Preds_ResNet50V2_wS_MC_ES_50epochs-1'
# savetitle2 = 'results/Plots/Prediction/Preds_ResNet50V2_wS_MC_ES_50epochs-2'

# model_path = r'results\samuel\ResNet50V2_nS_SC_100epochs\model.tf'
# savetitle1 = 'results/Plots/Prediction/Preds_ResNet50V2_nS_SC_100epochs-1'
# savetitle2 = 'results/Plots/Prediction/Preds_ResNet50V2_nS_SC_100epochs-2'

# model_path = r'results\samuel\ResNet50V2_nS_MC_100epochs\model.tf'
# savetitle1 = 'results/Plots/Prediction/Preds_ResNet50V2_nS_MC_100epochs-1'
# savetitle2 = 'results/Plots/Prediction/Preds_ResNet50V2_nS_MC_100epochs-2'


##################### PREPROCESS FUNCTION OF THE PRETRAINED ENCODER #####################

#from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess
#from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess
#from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess
preprocess_fcn = preprocess

onelabel = True


##################### NO CHANGES BELOW #####################

#DATA GENERATOR
img_dir = "PredImages/"
mask_dir = "PredLabels/"
image_extension = ".png"
mask_extension = ".png"
batch_size=3
horizontal_split = 12
vertical_split = 1
seed = 42
shift = False
single_img = False


train, validation = CustomDataGenerator.generate_data(batch_size, img_dir, mask_dir,horizontal_split, vertical_split, image_extension, mask_extension, 
                                                        preprocess_fcn, validation_split=0.1, flip=True, shift = shift, onelabel=onelabel, seed=seed, single_img=False)

model = tf.keras.models.load_model(model_path)
images, masks = train.__getitem__(8)
preds = model.predict(x=images, verbose=1)
np.save(savetitle1,np.array(preds[0]))
np.save(savetitle2,np.array(preds[1]))