#!/usr/bin/env python

import logging
import os.path
import sys
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import imagehash 		# for near duplicates
import hashlib

"""
Mock 1 Basketball Tracking Matrix - single Ball

tracker = Tracker( video_annotations_file=VIDEO_ANNOTATIONS_FILE )
frame_height, frame_width = tracker.shape
frame_path = tracker.frame_path(frame_number)

basketball_tracking_matrix = tracker.tracking_matrix(min_score=0.1)

	"frame", "x1", "x2", "y1", "y2", "radius", "center_x", "center_y", "overlap" - ints, boolean

NOTE: use numpy mask for missing datapoints as defualt
NOTE: use pandas for dataframes


video annotations data structures:

JSON / basic dict

# single image annotation info

	 	"image_hashes" : 
	 			{
	 				'md5' : MD5_HASH,
	 				'average_hash' : AVERAGE_HASH_PERCEPTUAL
	 			},
	 	"image_path",
	 	"image_dimensions",
	 	"detected_objects" : [

	 			{	
	 				"category" : "CATEGORY_NAME",			
	 				"score" : DETECTION_SCORE, 
					"bounding_box" : "x1", "x2", "y1", "y2",	
					"evalutation_model_used" : "MODEL_NAME"							# Tensorflow model used (or path)
				}, 
					...
			]

		#
		# Questions for Storing:
		#

		# 	- Tensorflow model information (path possibly)
		#	- Possibly Tensorflow Model Name Label Map
		# 	- Label Map used

ROWS:

	frame, md5_hash, average_hash, image_path, image_height, image_width, category, score, x1, x2, y1, y2, evaluation_model

"image_info_bundel" format:

			{
				"PATH/TO/FRAME/IMAGE" : 

						{

							"image_path" 		: "PATH/TO/FRAME/IMAGE",
							"image_folder" 		: "IMAGE_FOLDER"
							"image_filename" 	: "IMAGE_FILENAME",
							"image_height" 		: HEIGHT_IN_PICELS,
							"image_width" 		: WIDTH_IN_PICELS,
							"image_items_list" : 

								[
									"category" : "category_NAME",
									"score" : ACCURACY_SCORE.0,
									"box" : [x1,x2,y1,y2]
								]
						}
			}


"""

def image_hash(image_path, perceptual=False):
	"""
	:param image_path: path of image to hash
	:param perceptual: boolean, default False, use for near duplicate hash
	:return: image hash
	"""
	image = Image.open(image_path)
	if perceptual:
		return str(imagehash.average_hash(image))
	return str(hashlib.md5(image.tobytes()).hexdigest())

def frame_number(frame_path):
	"""
	format : 'frame_i.ext'
	:param frame_path: path to frame
	:return: frame number int
	"""
	#get filename
	filename = os.path.basename(frame_path)

	#strip extension
	filename_wout_ext = filename.split('.')[0]
	
	#frame_number
	frame = int(filename_wout_ext.split('_')[1])

	return frame

### testing

model_collection_name = "basketball_model_v1" 
shot_number = 16

#input video frames directory path
video_frames_dirpath = "/Users/ljbrown/Desktop/StatGeek/object_detection/video_frames/frames_shot_%s" % shot_number
frame_info_bundel_filepath = "/Users/ljbrown/Desktop/StatGeek/object_detection/%s/image_evaluator_output/shot_%s_image_info_bundel.json" % (model_collection_name,shot_number)

#output csv
output_csv_filepath = "out3.csv"


# load with pandas
frame_info_bundel = pd.read_json(frame_info_bundel_filepath)
column_names = ['frame', 'md5_hash', 'average_hash', 'image_path', 'image_width', 'image_height','category', 'score', 'x1', 'x2', 'y1', 'y2', 'evaluation_model']
video_ann = pd.DataFrame(columns=column_names)

for frame_path, frame_info in frame_info_bundel.items():
	frame = frame_number(frame_path)
	md5_hash = image_hash(frame_path)
	average_hash = image_hash(frame_path, perceptual=True)
	image_width, image_height = frame_info["image_width"], frame_info["image_height"] #Image.open(frame_path).size

	for detected_object in frame_info['image_items_list']:
		category = detected_object['class']
		score = float(detected_object['score'])
		x1, x2, y1, y2 = detected_object['box']

		# manual for this round
		evaluation_model = model_collection_name

		if (score > 15) and (category in ['person', 'basketball']):
			row = [frame, md5_hash, average_hash, frame_path, image_width, image_height, category, score, x1, x2, y1, y2, evaluation_model]
			pd_row = pd.DataFrame([row], columns=column_names)
			print(pd_row)
			video_ann = video_ann.append(pd_row)
		

video_ann.to_csv(output_csv_filepath)

