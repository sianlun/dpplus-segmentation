#!/usr/bin/env python
# coding: utf-8

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
ROOT_DIR = os.path.join(os.getcwd())

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib

print(ROOT_DIR)


# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR,"mask_rcnn_coco.h5")
# Directory to save logs and model checkpoints
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

print(COCO_WEIGHTS_PATH)
print(DEFAULT_LOGS_DIR)

class SegmentationConfig(Config):

  NAME = "deepspray"
  BACKBONE = "resnet101"

  # We use a GPU with 12GB memory, which can fit two images.
  IMAGES_PER_GPU = 1

  # Number of classes (including background)
  NUM_CLASSES = 1 + 4  # Background + dech-lig + drop + attchlig + lobe

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
    self.add_class("deepspray", 1, "detchlgm")
    self.add_class("deepspray", 2, "drop")
    self.add_class("deepspray", 3, "attchlgm")
    self.add_class("deepspray", 4, "lobe")

    # Train or validation dataset?
    assert subset in ["train", "valid"]

    if(subset == "train"):
      num_image_set = 1000
      label_file = os.path.join(dataset_dir, "labels_train_green_20_11_21.json")
    else:
      num_image_set = 200
      label_file = os.path.join(dataset_dir, "labels_valid_green_20_11_21.json")

    dataset_dir = os.path.join(dataset_dir, subset)
    annotations = json.load(open(label_file))

    for i in range (num_image_set):
      imgurl = os.path.join(dataset_dir, str(i)+".png")
      image = skimage.io.imread(imgurl)
      height, width = image.shape[:2]

      dc_from_json = annotations[i]
      polygons = []
      class_ids = []
      for j, p in enumerate(dc_from_json):
        p[1] = [j*height for j in p[1]]
        p[0] = [j*width for j in p[0]]
        polygons.append([p[0],p[1]])
        class_ids.append(p[2])
      self.add_image(
          "deepspray",
          image_id=str(i)+".png",  # use file name as a unique image id
          path=imgurl,
          width=width, height=height,
          polygons=polygons,
          class_ids=class_ids)

  def load_mask(self, image_id):
    """Generate instance masks for an image.
    Returns:
    masks: A bool array of shape [height, width, instance count] with
        one mask per instance.
    class_ids: a 1D array of class IDs of the instance masks.
    """
    # If not a balloon dataset image, delegate to parent class.
    image_info = self.image_info[image_id]
    if image_info["source"] != "deepspray":
        return super(self.__class__, self).load_mask(image_id)

    # Convert polygons to a bitmap mask of shape
    # [height, width, instance_count]
    info = self.image_info[image_id]
    class_ids = image_info['class_ids']
    mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
    # mask = np.zeros([info["height"]+1, info["width"]+1, len(info["polygons"])], dtype=np.uint8)
    for i, p in enumerate(info["polygons"]):
        # Get indexes of pixels inside the polygon and set them to 1
        rr, cc = skimage.draw.polygon(p[1], p[0])
        rr[rr > mask.shape[0]-1] = mask.shape[0]-1
        cc[cc > mask.shape[1]-1] = mask.shape[1]-1
        #print(rr, cc)
        mask[rr, cc, i] = 1

    # Return mask, and array of class IDs of each instance. Since we have
    # one class ID only, we return an array of 1s
    class_ids = np.array(class_ids, dtype=np.int32)
    return mask, class_ids #np.ones([mask.shape[-1]], dtype=np.int32)

  def image_reference(self, image_id):
    """Return the path of the image."""
    info = self.image_info[image_id]
    if info["source"] == "deepspray":
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
  print("Dataset ready")
  return dataset_train,dataset_val

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument("--backbone",
                        metavar="<backbone>",
                        help="'resnet50' or 'resnet101'")
    parser.add_argument("--epoch",
                        metavar="<backbone>",
                        help="number of epoch")
    parser.add_argument("--save_freq",
                        metavar="<backbone>",
                        help="Model saving frequency in terms of steps. Default is 10.",
                        default=10)
                        

    logs = DEFAULT_LOGS_DIR

    args = parser.parse_args()
    
    number_of_epoch = args.epoch

    config = SegmentationConfig()
    if args.backbone == 'resnet50':
      config.BACKBONE = "resnet50"
    else:
      config.BACKBONE = "resnet101"
    config.display()

    if args.command == "train":
      model = modellib.MaskRCNN(mode="training", config=config, model_dir=logs)
    else:
      model = modellib.MaskRCNN(mode="inference", config=config, model_dir=logs)

    if args.weights.lower() == "coco":
      weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        # if not os.path.exists(weights_path):
        #     utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
      #weights_path = model.find_last()[1]
      weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
      weights_path = model.get_imagenet_weights()
    else:
      weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    dataset_train,dataset_val = load_dataset_images("dataset_4_class")
    # saving_freq = int(config.STEPS_PER_EPOCH) * 2
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=int(number_of_epoch), layers='heads',saving_fre=int(args.save_freq))

    # this should be done. The trained weights are inside the logs directory. 

