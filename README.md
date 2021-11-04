# Segmentation for DeepSpray+ Project

This project uses code from https://github.com/leekunhee/Mask_RCNN @leekunhee that supports Tensorflow 2.x and keras 2.x. Tested on a Ubuntu 20.4 machine with RTX2070 running python 3.8

## Preparation
Apparently the Mask_RCNN works better with tf 1.13.1, and therefore python 3.6.x is required. The easiest way is to install python 3.6.x from deadsnake ppa
(refer to https://medium.com/analytics-vidhya/how-to-install-and-switch-between-different-python-versions-in-ubuntu-16-04-dc1726796b9b)
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.6
```

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
me@localhost: ~ $ mkdir myproject
me@localhost: ~ $ cd myproject
me@localhost: ~/myproject $ python3 -m virtualenv -p=/usr/bin/python3.6 venv3.6
me@localhost: ~/myproject $ source venv3.6/bin/activate
```
you should see something like the following:
```
(venv3.6) $ 
```

## clone project
```
(venv3.6) me@localhost: ~/myproject $ git clone https://github.com/sianlun/dpplus-segmentation.git
(venv3.6) me@localhost: ~/myproject $ cd dpplus-segmentation
```

## download h5 file
```
(venv3.6) me@localhost: ~/myproject/dpplus-segmentation $ wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/mask_rcnn_balloon.h5
```

## setup environment
```
(venv3.6) me@localhost: ~/myproject/dpplus-segmentation $ pip3 -r requirements.txt
```

you should be ready to go.



