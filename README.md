# Segmentation for DeepSpray+ Project

This project uses code from https://github.com/leekunhee/Mask_RCNN @leekunhee that supports Tensorflow 2.x and keras 2.x. Tested on a Ubuntu 20.4 machine with RTX2070 running python 3.8


## Ensure packages are up-to-date
```
$ sudo apt update
$ sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget liblzma-dev
```

## set up pip3 and virtualenv
```
$ pip3 install --upgrade pip
$ pip3 install virtualenv
```

## start working in the virtualenv
```
$ mkdir myproject
$ cd myproject
$ virtualenv venv
$ source venv/bin/activate
```
you should see something like the following:
```
(venv) $ 
```

## clone project
```
(venv) me@localhost: ~/myproject $ git clone https://github.com/sianlun/dpplus-segmentation.git
(venv) me@localhost: ~/myproject$ cd dpplus-segmentation
```

## download h5 file
```
(venv) me@localhost: ~/myproject/dpplus-segmentation $ wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/mask_rcnn_balloon.h5
```

## setup environment
```
(venv) me@localhost: ~/myproject/dpplus-segmentation $ pip3 -r requirements.txt
```

you should be ready to go.



