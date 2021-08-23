# mlmech

## Structure

### 1. Pre-trained Encoder simple Decoder

#### Encoder:
- VGG16
- MobileNet_v2
- ResNet50_v2

#### Decoder:
Very simple only `UpSampling2D` and `Conv2D`

### 2. Simple Segmentation Architectures
- U-Net
- SegNet
- FCN


### 3. Pre-trained Encoder and Advanced Decoder

Implement best Encoder in best Segmentation Model to get (hopefully) nice result

