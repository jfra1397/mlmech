# mlmech

## Structure

### 1. Pre-trained Encoder simple Decoder

#### Encoder:
- VGG16 (Julian)
- MobileNet_v2 (Lena)
- ResNet50_v2 (Samuel)
##### Test Case
- EarlyStopping/Shift = True <br>
  validation_split=0.01 <br>
  * Single Class (A)
  * Multiclass (B) <br>
  with 50 epochs
- EarlyStopping/Shift = False <br>
  validation_split=0.1 <br>
  * Single Class (C)
  * Multiclass (D) <br>
  with 100 epochs 

###### Results MobileNet_v2
 - A: results\lena\mobilenetV2\oneLabel\standard_V2
 - B: results\lena\mobilenetV2\multiLabel\standard
 - C: results\lena\mobilenetV2\oneLabel\withoutShift\noEarlyStopping\epochs100_validationSplit0-1
 - D: results\lena\mobilenetV2\multiLabel\withoutShift\noEarlyStopping_epochs100_validationSplit0-1
###### Results VGG16
- A: results/julian/vgg16_1
- B: results/julian/vgg16_3
- C: results/julian/vgg16_7
- D. results/julian/vgg16_6
#### Decoder:
Very simple only `UpSampling2D` and `Conv2D`

### Other Stuff:
#### VGG16:
- results/julian/vgg16_5 all trainable

### 2. Simple Segmentation Architectures
- U-Net (self build architecture) (Julian) <br>
  no shifts, early stopping applied (?), epochs=? <br>
  * Single Class: 
    * results/julian/unet_4; 256x256; epochs: 50; complexity: 4; EarlyStopping: false
    * results/julian/unet_256x3072; 256x3072; epochs: 20; complexity: 4; EarlyStopping: false
    * results/julian/unet_256x3072_2; 256x3072; epochs: 20; complexity: 3; EarlyStopping: false
    * results/julian/unet_256x3072_3; 256x3072; epochs: 20; complexity: 5; EarlyStopping: false
  * Multiclass
    * results/julian/unet_5; 256x256; epochs: 50; complexity: 5; EarlyStopping: false
    * results/julian/unet_256x3072_4; 256x3072; epochs: 50; complexity: 5; EarlyStopping: false
- SegNet (self build architecture) (Lena) <br>
  no shifts, early stopping applied (?), epochs=? <br>
  * Single Class
  * Multiclass
- FCN (Best Encoder with stronger Decoder) (Samuel)  <br>
  no shifts, early stopping applied (?), epochs=? <br>
  * Single Class
  * Multi class <br>
  Find out about the effect of `dropout`, `batchnormalization` and `General average pooling` <br>
  Analyse in use with a pretrained encoder, only decoder tuned.
  

### 3. Pre-trained Encoder and Advanced Decoder in advanced architecture

Implement the best (or the fastes?) encoder in the best architecture (Unet/Segnet/FCN) with a stronger decoder.


## Compare
 - Loss
 - Benchmark
 - Validation Loss
 - Opitcal Result

## On Plus
 - Big sized picture used in a self build architecture
 - Use a front-net in front of a pretrained encoder in order <br>
   to use the real picture size and also adjust the decoder for <br>
   the real sized pictures. --> More parameters to learn with!

