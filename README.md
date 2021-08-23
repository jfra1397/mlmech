# mlmech

## Structure

### 1. Pre-trained Encoder simple Decoder

#### Encoder:
- VGG16 (Julian)
- MobileNet_v2 (Lena)
- ResNet50_v2 (Samuel)

#### Decoder:
Very simple only `UpSampling2D` and `Conv2D`

### 2. Simple Segmentation Architectures
- U-Net (Julian)
- SegNet (Lena)
- FCN (Samuel)


### 3. Pre-trained Encoder and Advanced Decoder

Implement best Encoder in best Segmentation Model to get (hopefully) nice result


## Compare
 - Loss
 - Benchmark
 - Validation Loss
 - Opitcal Result

