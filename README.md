# Segmentation for DeepSpray+ Project

This project uses code from https://github.com/leekunhee/Mask_RCNN @leekunhee that supports Tensorflow 2.x and keras 2.x. Tested on a Ubuntu 20.4 machine with RTX2070 running python 3.8

## Preparation
Apparently the Mask_RCNN works better with tf 1.13.1, and therefore python 3.6.x is required. But we were able to update the code so now its executable with tf 2.6 and python 3.8.

## First, Ensure packages are up-to-date
```
$ sudo apt update
$ sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget liblzma-dev
```

## start working in the built in python
```
me@localhost: ~ $ mkdir myproject
me@localhost: ~ $ cd myproject
```

## clone project
```
me@localhost: ~/myproject $ git clone -b hpc https://github.com/sianlun/dpplus-segmentation.git
me@localhost: ~/myproject $ cd dpplus-segmentation
```

## download h5 file
```
me@localhost: ~/myproject/dpplus-segmentation $ wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/mask_rcnn_balloon.h5
```


## run the project with resnet50

```
me@localhost: ~/myproject/dpplus-segmentation $ python train_for_segmentation.py train --weight=coco --backbone=resnet50 --epoch=10000
```
After the 10000 epoch, please sent us the 1000th, 2000th, 3000th, 4000th, 5000th, 6000th, 7000th, 8000th, 9000th, 10000th weight file.
Weight files are located under log directory. 

## run the project with resnet101
```
me@localhost: ~/myproject/dpplus-segmentation $ python train_for_segmentation.py train --weight=coco --backbone=resnet101 --epoch=10000
```
After the 10000 epoch, please sent us the 1000th, 2000th, 3000th, 4000th, 5000th, 6000th, 7000th, 8000th, 9000th, 10000th weight file.
Weight files are located under log directory. 


