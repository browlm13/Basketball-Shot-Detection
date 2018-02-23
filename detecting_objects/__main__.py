#!/usr/bin/env python

# internal
import logging
import os
import glob
import cv2
import json

# my lib
from image_evaluator.src import image_evaluator

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

#saving image_evaluator evaluations
def save_image_directory_evaluations(image_directory_dirpath, image_boolean_bundel_filepath, image_info_bundel_filepath, model_list, bool_rule):

	#create image evaluator and load models 
	ie = image_evaluator.Image_Evaluator()
	ie.load_models(model_list)

	# get path to each frame in video frames directory
	image_path_list = glob.glob(image_directory_dirpath + "/*")
	
	#evaluate images in directory and write image_boolean_bundel and image_info_bundel to files for quick access
	image_boolean_bundel, image_info_bundel = ie.boolean_image_evaluation(image_path_list, bool_rule)

	with open(image_boolean_bundel_filepath, 'w') as file:
		file.write(json.dumps(image_boolean_bundel))

	with open(image_info_bundel_filepath, 'w') as file:
		file.write(json.dumps(image_info_bundel))

#loading saved evaluations
def load_image_info_bundel(image_info_bundel_filepath):
	with open(image_info_bundel_filepath) as json_data:
		d = json.load(json_data)
	return d

if __name__ == '__main__':

	#
	# Initial Evaluation
	#

	#video frames diretory (basketball_225.JPEG - basketball_323.JPEG)
	video_frames_dirpath = "/Users/ljbrown/Desktop/StatGeek/object_detection/video_frames/"

	#image_boolean_bundel and image_info_bundel file paths for quick access
	image_boolean_bundel_filepath = "/Users/ljbrown/Desktop/StatGeek/object_detection/image_evaluator_output/image_boolean_bundel.json"
	image_info_bundel_filepath = "/Users/ljbrown/Desktop/StatGeek/object_detection/image_evaluator_output/image_info_bundel.json"

	#tensorflow models
	BASKETBALL_MODEL = {'name' : 'basketball_model_v1', 'use_display_name' : False, 'paths' : {'frozen graph': "image_evaluator/models/basketball_model_v1/frozen_inference_graph/frozen_inference_graph.pb", 'labels' : "image_evaluator/models/basketball_model_v1/label_map.pbtxt"}}
	PERSON_MODEL = {'name' : 'ssd_mobilenet_v1_coco_2017_11_17', 'use_display_name' : True, 'paths' : {'frozen graph': "image_evaluator/models/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph/frozen_inference_graph.pb", 'labels' : "image_evaluator/models/ssd_mobilenet_v1_coco_2017_11_17/mscoco_label_map.pbtxt"}}

	#bool rule - any basketball or person above an accuracy score of 40.0
	bool_rule = "any('basketball', 40.0) or any('person', 40.0)"

	#save to files for quick access
	#save_image_directory_evaluations(video_frames_dirpath, image_boolean_bundel_filepath, image_info_bundel_filepath, [BASKETBALL_MODEL, PERSON_MODEL], bool_rule)
	
	#load saved image_info_bundel
	image_info_bundel = load_image_info_bundel(image_info_bundel_filepath)
	


