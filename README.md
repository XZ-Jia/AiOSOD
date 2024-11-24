# All in One: RGB, RGB-D and RGB-T Salient Object Detection

Performing RGB, RGB-D and RGB-T saliency detection requires only needs one model with one weight file.

The details are in https://arxiv.org/abs/2311.14746.

## Requirement
1. Python 3.8
2. Pytorch 2.1.2
3. Torchvison 0.7.0

Your folder should look like this:
````
-- Data
   |-- NJUD
   |   |-- trainset
   |   |-- | RGB
   |   |-- | depth
   |   |-- | GT
   |   |-- testset
   |   |-- | RGB
   |   |-- | depth
   |   |-- | GT
   |-- LFSD
   |   |-- | RGB
   |   |-- | depth
   |   |-- | GT
   |-- VT5000
   |   |-- trainset
   |   |-- | RGB
   |   |-- | depth
   |   |-- | GT
   |   |-- testset
   |   |-- | RGB
   |   |-- | depth
   |   |-- | GT
   |-- VT1000
   |   |-- | RGB
   |   |-- | depth
   |   |-- | GT
   |-- ECSSD
   |   |-- | RGB
   |   |-- | depth
   |   |-- | GT
   ...
-- t2t-vit
-- swin
-- pvt
````
Note: The depth folder in the RGB dataset has the same content as the RGB folder, while the depth folder in the RGB-T dataset contains thermal images.


### RGB, RGB-D, RGB-T Saliency Maps 
The saliency maps in our paper can be downloaded from  [[Google drive](https://drive.google.com/file/d/14d2HHzW-THZNA35iYbg7uU3vHq_F2Grd/view?usp=sharing)].



### Training, and Testing
1. Download the pre-trained models [[T2T-ViT-10](https://github.com/yitu-opensource/T2T-ViT)],[[Swin-tiny](https://github.com/microsoft/Swin-Transformer)],[[PVTv2-B0](https://github.com/whai362/PVT)] to the t2t-vit, swin, pvt folder, respectively.
2. Modify settings such as the path of main.py. Run `python main.py --Training True --Testing True` for training, and testing. 

