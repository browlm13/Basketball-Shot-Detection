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

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

def box_area(box):
	"""
	:param box: tuple (x1,x2,y1,y2)
	:returns: area of box
	"""
	(left, right, top, bottom) = box
	return (right-left) * (bottom-top)

def iou(box1, box2):
	"""
	:param box1: tuple (x1,x2,y1,y2)
	:param box2: tuple (x1,x2,y1,y2)
	:return: Intersection over union" of two bounding boxes as float (0,1)
	"""

	#return "intersection over union" of two bounding boxes as float (0,1)
	paired_boxes = tuple(zip(box1, box2))

	# find intersecting box
	intersecting_box = (max(paired_boxes[0]), min(paired_boxes[1]), max(paired_boxes[2]), min(paired_boxes[3]))

	# adjust for min functions
	if (intersecting_box[1] < intersecting_box[0]) or (intersecting_box[3] < intersecting_box[2]):
		return 0.0

	# compute the intersection over union
	return box_area(intersecting_box)  / float( box_area(box1) + box_area(box2) - box_area(intersecting_box) )


def read_shot_info_matrix(file_path):
	# read csv file
	tracking_matrix = pd.read_csv(file_path)

	# ignore score, image path, md5 hash and average hash

	# change frame from float to int
	tracking_matrix['frame'] = tracking_matrix['frame'].astype("int")

	# creat full frame dataframe and remove duplicates
	frames = tracking_matrix[['frame']]
	total_frames_df = frames.drop_duplicates(subset=['frame'], keep='first').sort('frame')

	# initial annotation object columns
	retrieved_columns=['frame', 'x1', 'x2', 'y1', 'y2']

	#made need to ensure only one bb or player is chosen for each frame
	basketball_tracking_matrix = tracking_matrix.loc[tracking_matrix['category'] == "basketball"][retrieved_columns].sort('frame', ascending=True)
	person_tracking_matrix = tracking_matrix.loc[tracking_matrix['category'] == "person"][retrieved_columns].sort('frame', ascending=True)

	# retrieve basketball and person indepented dfs
	# inner join for iou calculation
	basketball_columns = ['frame', 'bx1', 'bx2', 'by1', 'by2']
	basketball_columns = ['frame', 'px1', 'px2', 'py1', 'py2']
	basketball_tracking_matrix.columns = basketball_columns
	person_tracking_matrix.columns = basketball_columns

	inner_bp_tracking_matrix = pd.merge(basketball_tracking_matrix,person_tracking_matrix, how='inner', on=["frame"])

	iou_dict = {'frame': [], 'iou': []}
	for tuple_row in inner_bp_tracking_matrix.itertuples(index=False, name=None):
		iou_frame, bbox, pbox = tuple_row[0], tuple_row[1:5], tuple_row[5:]
		iou_val = iou(bbox, pbox)
		iou_dict['frame'].append(iou_frame)
		iou_dict['iou'].append(iou_val)
	
	iou_matrix = pd.DataFrame(iou_dict)

	# calculate ball radii
	

	# join all dataframes
	boxes_iou_matrix = pd.merge(inner_bp_tracking_matrix, iou_matrix, how='left', on=['frame'])
	frame_ball_tracking_matrix = pd.merge(total_frames_df, boxes_iou_matrix, how='outer', on=['frame'])
	
	# set frame as index
	frame_ball_tracking_matrix.set_index(['frame'], inplace=True)
	print(frame_ball_tracking_matrix)


### testing

model_collection_name = "basketball_model_v1" 

shots = [2] #range(1,4)
for shot_number in shots:

	#shot_number = 2
	MIN_SCORE = 25

	#input video frames directory path
	video_frames_dirpath = "/Users/ljbrown/Desktop/StatGeek/object_detection/video_frames/frames_shot_%s" % shot_number
	video_frames_dirpath = "/Users/ljbrown/Desktop/StatGeek/object_detection/video_frames/frames_shot_%s" % shot_number
	frame_info_bundel_filepath = "/Users/ljbrown/Desktop/StatGeek/object_detection/%s/image_evaluator_output/shot_%s_image_info_bundel.json" % (model_collection_name,shot_number)
	csv_video_frame_output_file_path = "/Users/ljbrown/Desktop/StatGeek/object_detection/%s/image_evaluator_output/shot_%s_frame_info_matrix.csv" % (model_collection_name,shot_number)

	#output csv
	output_csv_filepath = csv_video_frame_output_file_path

	"""
	# load with pandas
	frame_info_bundel = pd.read_json(frame_info_bundel_filepath)
	column_names = ['frame', 'md5_hash', 'average_hash', 'image_path', 'image_width', 'image_height','category', 'score', 'x1', 'x2', 'y1', 'y2', 'evaluation_model']
	video_ann = pd.DataFrame(columns=column_names)

	for frame_path, frame_info in frame_info_bundel.items():
		frame = int(frame_number(frame_path))
		md5_hash = image_hash(frame_path)
		average_hash = image_hash(frame_path, perceptual=True)
		image_width, image_height = frame_info["image_width"], frame_info["image_height"] #Image.open(frame_path).size

		for detected_object in frame_info['image_items_list']:
			category = detected_object['class']
			score = float(detected_object['score'])
			x1, x2, y1, y2 = detected_object['box']

			# manual for this round
			evaluation_model = model_collection_name

			if (score > MIN_SCORE) and (category in ['person', 'basketball']):
				row = [frame, md5_hash, average_hash, frame_path, image_width, image_height, category, score, x1, x2, y1, y2, evaluation_model]
				pd_row = pd.DataFrame([row], columns=column_names)
				print(pd_row)
				video_ann = video_ann.append(pd_row)
				

	video_ann.to_csv(output_csv_filepath)
	"""

for shot_number in shots:

	logger.info("logging shot number %d" % shot_number)
	input_csv_filepath = csv_video_frame_output_file_path = "/Users/ljbrown/Desktop/StatGeek/object_detection/%s/image_evaluator_output/shot_%s_frame_info_matrix.csv" % (model_collection_name,shot_number)
	read_shot_info_matrix(input_csv_filepath)




