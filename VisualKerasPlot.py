### NEEDED FOR THIS CODE EXECUTION ###
# pip3 install pydotplus
# pip3 install graphviz
# pip3 install pydot

import tensorflow as tf
import visualkeras
from PIL import ImageFont
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, UpSampling2D
from tensorflow.python.keras.layers import ZeroPadding2D, BatchNormalization, Add, Activation
from collections import defaultdict


#LENA
#model_path = r'results\lena\mobilenetV2\oneLabel\standard_V2\model.tf'
#savetitle = 'results/Plots/Model/MobileNetV2.png'

#JULIAN
#model_path = r'results\julian\vgg16_1\model.tf'
#savetitle = 'results/Plots/Model/VGG16.png'

#SAMUEL
model_path = r'results\samuel\ResNet50V2_wS_SC_ES_50epochs\model.tf'
savetitle = 'results/Plots/Model/ResNet50V2.png'

model = tf.keras.models.load_model(model_path)

color_map = defaultdict(dict)
color_map[MaxPooling2D]['fill'] = 'fuchsia'
color_map[Conv2D]['fill'] = 'blue'
color_map[BatchNormalization]['fill'] = 'lightgray' #Wird ignoriert, warum auch immer. Kann auch nicht im type_ignore berücksichtigt werden
color_map[UpSampling2D]['fill'] = 'fuchsia'
color_map[Activation]['fill'] = 'gray'
color_map[Add]['fill'] = 'crimson'
color_map[Dropout]['fill'] = 'black'

font = ImageFont.truetype("arial.ttf", 50)  # using comic sans is strictly prohibited!
visualkeras.layered_view(model, to_file = savetitle, type_ignore=[ZeroPadding2D, Flatten], 
                            color_map=color_map, legend=True, font=font).show()