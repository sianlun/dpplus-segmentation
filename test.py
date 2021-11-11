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

COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR,"mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
print(ROOT_DIR)

class TestConfig(Config):
  NAME = "deepspray-test"
  IMAGES_PER_GPU = 1

  # Number of classes (including background)
  NUM_CLASSES = 1 + 1  # Background + baloon

def color_splash(image, mask):
  """Apply color splash effect.
  image: RGB image [height, width, 3]
  mask: instance segmentation mask [height, width, instance count]
  Returns result image.
  """
  # Make a grayscale copy of the image. The grayscale copy still
  # has 3 RGB channels, though.
  gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
  # We're treating all instances as one, so collapse the mask into one layer
  mask = (np.sum(mask, -1, keepdims=True) >= 1)
  # Copy color pixels from the original color image where mask is set
  if mask.shape[0] > 0:
      splash = np.where(mask, image, gray).astype(np.uint8)
  else:
      splash = gray
  return cv2.cvtColor(splash, cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--image', required=True,
                        metavar="/path/to/image.png",
                        help="Path to image .png file")
                      
    args = parser.parse_args()
    imagePath = args.image
    _, imageName = os.path.split(imagePath)
    args = parser.parse_args()
    config = TestConfig()
    config.display()
    command = "test"
    logs = DEFAULT_LOGS_DIR
    # weights = "coco"

    print(imagePath)
    print(imageName)


    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=logs)

    if args.weights.lower() == "coco":
      weights_path = COCO_WEIGHTS_PATH
    else:
      weights_path = args.weights
      # weights_path = "/content/drive/MyDrive/weights/refat-01/mask_rcnn_deepspray_0100.h5"

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

    image1 = skimage.io.imread(imagePath)
    r = model.detect([image1], verbose=1)
    splash = color_splash(image1, r[0]['masks'])
    cv2.imwrite(imageName,splash)
    print("Done!")
