#source: https://github.com/tensorflow/models/blob/master/research/object_detection/

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2
import PIL.Image as Image

# This is needed since the file is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

#
# Model preparation 
#

# Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

#
# Download Model
#

"""
# In[5]:

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
"""

#
# Load a (frozen) Tensorflow model into memory.
#

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

#
# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
#

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#
# Helper code
#

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def image_dimensions(image_np):
  image_pil = Image.fromarray(np.uint8(image_np)).convert('RGB')
  return image_pil.size

#path, folder, filename
def get_path_data(path):
  folder = os.path.basename(os.path.dirname(path))
  filename = os.path.basename(path)
  return path, folder, filename


def selected_items_list(max_boxes_to_draw, min_score_thresh, selected_classes, image_np, boxes, scores, classes):
      # retrieve image size
      image_pil = Image.fromarray(np.uint8(image_np)).convert('RGB')
      im_width, im_height = image_pil.size

      #box, class, score
      selected_items = []

      for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:

          item = {}

          #
          # box
          #
          normalized_box = tuple(boxes[i].tolist())
          n_ymin, n_xmin, n_ymax, n_xmax = normalized_box
          box = (int(n_xmin * im_width), int(n_xmax * im_width), int(n_ymin * im_height), int(n_ymax * im_height)) #(left, right, top, bottom)
          item['box'] = box

          #test coors
          #(left, right, top, bottom) = box
          #cv2.rectangle(image_np,(left,top),(right,bottom),(0,255,0),3)

          #
          # class name
          #
          class_name = 'N/A'
          if classes[i] in category_index.keys(): class_name = str(category_index[classes[i]]['name'])

          item['class'] = class_name
          
          #
          # detection score
          #
          item['score'] = 100*scores[i]

          # add if class is in selected_classes
          if item['class'] in selected_classes:
            selected_items.append(item)

      return selected_items

#
# crete image data sict with properties for xml annotation: folder, filename, path, width, height, depth(3), objects: class/name, xmin, ymin, xmax, ymax
#

def get_image_data(selected_items, image_np, image_path):

  path, folder, filename = get_path_data(image_path)
  width, height = image_dimensions(image_np)
  depth = 3

  image_data = {}
  image_data['path'] = path
  image_data['folder'] = folder
  image_data['filename'] = filename
  image_data['width'] = width
  image_data['height'] = height
  image_data['depth'] = depth

  image_data['objects'] = []
  for item in selected_items:
    o = {}
    o['name'] = item['class']

    xmin, xmax, ymin, ymax = item['box']
    o['xmin'] = xmin
    o['ymin'] = ymin
    o['xmax'] = xmax
    o['ymax'] = ymax

    image_data['objects'].append(o)

  return image_data

#
# Test accuracy by writing new images
#

def write_image_for_accuracy_test(output_directory_path, image_file_name, image_np, selected_items):

  for item in selected_items:

    #test coors acuracy
    (left, right, top, bottom) = item['box']
    cv2.rectangle(image_np,(left,top),(right,bottom),(0,255,0),3)

  #write
  output_file = os.path.join(output_directory_path, image_file_name)
  cv2.imwrite(output_file, image_np)

#
# Images
#

# Input images

# add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'basketball_{}.JPEG'.format(i)) for i in range(0, 1) ]
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image2.jpg')]

# Output images
PATH_TO_TEST_IMAGES_ACCURACY_DIR = 'check_image_accuracy'

#
# Settings
#

max_boxes_to_draw = 100
min_score_thresh = 0.6
selected_classes = ['person', 'sports ball']


#
# Detection
#

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')             # Definite input and output Tensors for detection_graph
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')       # Each box represents a part of the image where a particular object was detected.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')     # Each score represent how level of confidence for each of the objects.
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')


    #
    # Image Detection Loop
    #

    for image_path in TEST_IMAGE_PATHS:

      #
      # meta data
      #

      image_path, image_folder, image_filename = get_path_data(image_path)

      #
      # prepare image for model input
      #

      image = Image.open(image_path)
      image_np = load_image_into_numpy_array(image)
      image_np_expanded = np.expand_dims(image_np, axis=0) # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      
      #
      # Detection
      #

      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})

      #
      # Reformat results
      #

      boxes = np.squeeze(boxes)
      scores = np.squeeze(scores)
      classes = np.squeeze(classes).astype(np.int32)

      #
      # Get selected items (box, class, score)
      #

      selected_items = selected_items_list(max_boxes_to_draw, min_score_thresh, selected_classes, image_np, boxes, scores, classes)

      #
      # Properties for xml format
      #

      image_data = get_image_data(selected_items, image_np, image_path)

      #
      # Test accuracy by writing new images
      #

      write_image_for_accuracy_test(PATH_TO_TEST_IMAGES_ACCURACY_DIR, image_filename, image_np, selected_items)


    

