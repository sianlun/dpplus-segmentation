#!/usr/bin/env python

import os
import sys
import json
import datetime
import skimage.io
import numpy as np
import skimage.draw
import cv2
import matplotlib.pyplot as plt
from PIL import Image 
import glob
import random, csv, json
from scipy import ndimage
import argparse
import warnings
warnings.filterwarnings('ignore')


# Root directory of the project
ROOT_DIR = os.path.join(os.getcwd(),"./")
from config import Config
import utils
import model as modellib

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Directory to save logs and model checkpoints
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

#print(COCO_WEIGHTS_PATH)
#print(DEFAULT_LOGS_DIR)

class SegmentationConfig(Config):
  """Configuration for training on the toy  dataset.
  Derives from the base Config class and overrides some values.
  """
  # Give the configuration a recognizable name
  NAME = "deepspray"

  # We use a GPU with 12GB memory, which can fit two images.
  # Adjust down if you use a smaller GPU.
  IMAGES_PER_GPU = 2

  # Number of classes (including background)
  NUM_CLASSES = 1 + 1  # Background + dech-lig

  # Number of training steps per epoch
  STEPS_PER_EPOCH = 100

  # Skip detections with < 90% confidence
  DETECTION_MIN_CONFIDENCE = 0.9

############################################################
#  Dataset
############################################################

class SegmentationDataset(utils.Dataset):

  def load_dataset(self, dataset_dir, subset):
    """Load a subset of the deepspray dataset.
    dataset_dir: Root directory of the dataset.
    subset: Subset to load: train or val
    """
    # Add classes. We have only one class to add.
    self.add_class("detchlgm", 1, "detchlgm")

    # Train or validation dataset?
    assert subset in ["train", "valid"]

    if(subset == "train"):
      num_image_set = 1000
      label_file = os.path.join(dataset_dir, "labels_train_green_30_10_2021.json")
    else:
      num_image_set = 200
      label_file = os.path.join(dataset_dir, "labels_valid_green_30_10_2021.json")

    dataset_dir = os.path.join(dataset_dir, subset)
    annotations = json.load(open(label_file))

    for i in range (num_image_set):
      imgurl = os.path.join(dataset_dir, str(i)+".png")
      image = skimage.io.imread(imgurl)
      height, width = image.shape[:2]

      dc_from_json = annotations[i]
      polygons = []
      for j, p in enumerate(dc_from_json):
        p[1] = [j*height for j in p[1]]
        p[0] = [j*width for j in p[0]]
        polygons.append([p[0],p[1]])

      self.add_image(
          "detchlgm",
          image_id=str(i)+".png",  # use file name as a unique image id
          path=imgurl,
          width=width, height=height,
          polygons=polygons)

  def load_mask(self, image_id):
    """Generate instance masks for an image.
    Returns:
    masks: A bool array of shape [height, width, instance count] with
        one mask per instance.
    class_ids: a 1D array of class IDs of the instance masks.
    """
    # If not a balloon dataset image, delegate to parent class.
    image_info = self.image_info[image_id]
    if image_info["source"] != "detchlgm":
        return super(self.__class__, self).load_mask(image_id)

    # Convert polygons to a bitmap mask of shape
    # [height, width, instance_count]
    info = self.image_info[image_id]
    mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                    dtype=np.uint8)
    for i, p in enumerate(info["polygons"]):
        # Get indexes of pixels inside the polygon and set them to 1
        rr, cc = skimage.draw.polygon(p[1], p[0])
        mask[rr, cc, i] = 1

    # Return mask, and array of class IDs of each instance. Since we have
    # one class ID only, we return an array of 1s
    return mask, np.ones([mask.shape[-1]], dtype=np.int32)

  def image_reference(self, image_id):
    """Return the path of the image."""
    info = self.image_info[image_id]
    if info["source"] == "detchlgm":
        return info["path"]
    else:
        super(self.__class__, self).image_reference(image_id)

def load_dataset_images(dataset):
  # Train or evaluate
  dataset_train = SegmentationDataset()
  dataset_train.load_dataset(dataset, "train")
  dataset_train.prepare()

  # Validation dataset
  dataset_val = SegmentationDataset()
  dataset_val.load_dataset(dataset, "valid")
  dataset_val.prepare()
  return dataset_train,dataset_val

config = SegmentationConfig()
config.display()
command = "train"
logs = DEFAULT_LOGS_DIR
weights = "coco"

if command == "train":
  model = modellib.MaskRCNN(mode="training", config=config, model_dir=logs)
else:
  model = modellib.MaskRCNN(mode="inference", config=config, model_dir=logs)

if weights.lower() == "coco":
  weights_path = COCO_WEIGHTS_PATH
    # Download weights file
    # if not os.path.exists(weights_path):
    #     utils.download_trained_weights(weights_path)
elif weights.lower() == "last":
    # Find last trained weights
  weights_path = model.find_last()[1]
elif weights.lower() == "imagenet":
    # Start from ImageNet trained weights
  weights_path = model.get_imagenet_weights()
else:
  weights_path = weights

# Load weights
print("Loading weights ", weights_path)
if weights.lower() == "coco":
    # Exclude the last layers because they require a matching
    # number of classes
    model.load_weights(weights_path, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])
else:
    model.load_weights(weights_path, by_name=True)


dataset_train,dataset_val = load_dataset_images("dataset")

model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')

# this should be done. The trained weights are inside the logs directory. 

