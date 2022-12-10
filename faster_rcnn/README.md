# Faster R-CNN

## This project is mainly from pytorch torchvision 
* https://github.com/pytorch/vision/tree/master/torchvision/models/detection


## Instruction：
1. Environment: See *requirements.txt*
2. Download pretrained weight to backbone:
  - ResNet50+FPN backbone: https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth
3. Dataset
  - Pascal VOC: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar 
4. Train on single GPU:
  - for Resnet50+fpn backbone: *python train.py*
  - for Mobilenetv2 backbone *python train.py*
