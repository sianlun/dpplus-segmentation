# Segmentation for DeepSpray+ Project

This project uses code from https://github.com/leekunhee/Mask_RCNN @leekunhee that supports Tensorflow 2.x and keras 2.x. Tested on a Ubuntu 20.4 machine with RTX2070 running python 3.8

## Preparation
Apparently the Mask_RCNN works better with tf 1.13.1, and therefore python 3.6.x is required. But we were able to update the code so now its executable with tf 2.6 and python 3.8.

## start working with python 3.8 inside a directory
```
me@localhost: ~ $ conda create --prefix /home/user/deepspray_segmentation python=3.8
me@localhost: ~ $ conda activate /home/user/deepspray_segmentation
(/home/user/deepspray_segmentation) me@localhost: ~ $ cd deepspray_segmentation
```

## clone project
```
(/home/user/deepspray_segmentation) me@localhost: ~/deepspray_segmentation $ conda install git
(/home/user/deepspray_segmentation) me@localhost: ~/deepspray_segmentation $ git clone -b hpc https://github.com/sianlun/dpplus-segmentation.git
(/home/user/deepspray_segmentation) me@localhost: ~/deepspray_segmentation $ cd dpplus-segmentation

```

## download h5 file
```
(/home/user/deepspray_segmentation) me@localhost: ~/deepspray_segmentation/dpplus-segmentation $ wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
```

## install libraries
```
(/home/user/deepspray_segmentation) me@localhost: ~/deepspray_segmentation/dpplus-segmentation $ conda install -c conda-forge tensorflow=2.6
(/home/user/deepspray_segmentation) me@localhost: ~/deepspray_segmentation/dpplus-segmentation $ pip install -U scikit-image==0.16.2
(/home/user/deepspray_segmentation) me@localhost: ~/deepspray_segmentation/dpplus-segmentation $ pip install opencv-python
(/home/user/deepspray_segmentation) me@localhost: ~/deepspray_segmentation/dpplus-segmentation $ pip install matplotlib
```

## run the project with resnet50

```
(/home/user/deepspray_segmentation) me@localhost: ~/deepspray_segmentation/dpplus-segmentation $ python train_for_segmentation.py train --weight=coco --backbone=resnet50 --epoch=10000
```
After the 10000 epoch, please sent us the 1000th, 2000th, 3000th, 4000th, 5000th, 6000th, 7000th, 8000th, 9000th, 10000th weight file.
Weight files are located under log directory. 

## run the project with resnet101
```
(/home/user/deepspray_segmentation) me@localhost: ~/deepspray_segmentation/dpplus-segmentation $ python train_for_segmentation.py train --weight=coco --backbone=resnet101 --epoch=10000
```
After the 10000 epoch, please sent us the 1000th, 2000th, 3000th, 4000th, 5000th, 6000th, 7000th, 8000th, 9000th, 10000th weight file.
Weight files are located under log directory. 


