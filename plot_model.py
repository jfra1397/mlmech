
import tensorflow as tf
model_path = "results/julian/vgg16_1/model.tf"
model = tf.keras.models.load_model(model_path)
import visualkeras
from PIL import ImageFont
font = ImageFont.truetype("arial.ttf", 50)  # using comic sans is strictly prohibited!
visualkeras.layered_view(model, legend=True, font=font).show()