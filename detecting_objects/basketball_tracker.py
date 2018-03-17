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


def get_box_center_point(box):
	"""
	:param ball_box: tupple (x1,x2,y1,y2)
	:returns: Point (x,y) where x and y are half the box's width and height
	"""
	# 1/2 height, 1/2 width
	(left, right, top, bottom) = box
	width = int((right - left)/2)
	x = left + width
	height = int((bottom - top)/2)
	y = top + height
	return (x,y)


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

def get_ball_radius(ball_box, integer=True):
	"""
	:param ball_box: tupple (x1,x2,y1,y2)
	:param integer: boolean, True by defualt
	:returns: average between half the box's width and half the box's height, defualt returns int
	"""
	(left, right, top, bottom) = ball_box
	xwidth = (right - left)/2
	ywidth = (bottom - top)/2
	radius = (xwidth + ywidth)/2

	if integer: return int(radius)
	return radius


#use: matrix_with_basketball_boxes.apply(get_basketball_box_center_dataframe, axis=1)
def get_basketball_box_center_dataframe(matrix_with_basketball_boxes):
	box = matrix_with_basketball_boxes.loc[["x1_basketball", 'x2_basketball', 'y1_basketball', 'y2_basketball']]
	frame = matrix_with_basketball_boxes.loc["frame"]
	if box.isnull().values.any():
		x, y =  np.nan, np.nan
	else:
		x, y = get_box_center_point(box)

	return pd.Series(data=[frame,x,y], index=['frame','x_basketball', 'y_basketball'])

#use: add_radii_column(matrix_with_bbox)
def add_basketball_box_center_column(matrix_with_basketball_boxes):
	df = matrix_with_basketball_boxes.apply(get_basketball_box_center_dataframe,axis=1)
	return pd.merge(matrix_with_basketball_boxes,df, on=['frame'], how='outer') #df


#use: matrix_with_bbox_pbox.apply(get_iou_dataframe, axis=1)
# 1 for person ball intersecting bounding boxes, 0 for no overlap
def get_free_dataframe(matrix_with_basketballand_person_boxes):
	bbox = matrix_with_basketballand_person_boxes.loc[["x1_basketball", 'x2_basketball', 'y1_basketball', 'y2_basketball']]
	pbox = matrix_with_basketballand_person_boxes.loc[["x1_person", 'x2_person', 'y1_person', 'y2_person']]
	if (bbox.isnull().values.any()) or (pbox.isnull().values.any()):
		return np.nan
	elif iou(bbox, pbox) > 0:
		return 1
	return 0	

#use: add_free_column(matrix_with_bbox_pbox)
def add_free_column(matrix_with_basketballand_person_boxes):
	return matrix_with_basketballand_person_boxes.assign(free = matrix_with_basketballand_person_boxes.apply(get_free_dataframe,axis=1).to_frame())


#use: matrix_with_bbox_pbox.apply(get_iou_dataframe, axis=1)
def get_iou_dataframe(matrix_with_basketballand_person_boxes):
	bbox = matrix_with_basketballand_person_boxes.loc[["x1_basketball", 'x2_basketball', 'y1_basketball', 'y2_basketball']]
	pbox = matrix_with_basketballand_person_boxes.loc[["x1_person", 'x2_person', 'y1_person', 'y2_person']]
	if (bbox.isnull().values.any()) or (pbox.isnull().values.any()) :
		return np.nan
	return iou(bbox, pbox)

#use: add_iou_column(matrix_with_bbox_pbox)
def add_iou_column(matrix_with_basketballand_person_boxes):
	return matrix_with_basketballand_person_boxes.assign(iou = matrix_with_basketballand_person_boxes.apply(get_iou_dataframe,axis=1).to_frame())

#use: matrix_with_bbox.apply(get_radii_dataframe, axis=1)
def get_radii_dataframe(matrix_with_basketball_boxes):
	box = matrix_with_basketball_boxes.loc[["x1_basketball", 'x2_basketball', 'y1_basketball', 'y2_basketball']]
	if box.isnull().values.any():
		return np.nan
	return get_ball_radius(box, integer=False)

#use: add_radii_column(matrix_with_bbox)
def add_radii_column(matrix_with_basketball_boxes):
	return matrix_with_basketball_boxes.assign(radius = matrix_with_basketball_boxes.apply(get_radii_dataframe,axis=1).to_frame())

def read_shot_info_matrix(file_path):

	# read csv file
	tracking_matrix = pd.read_csv(file_path)

	# filter tracking matrix to used datapoints only
	tracking_matrix= tracking_matrix[['frame','score','category', 'x1','x2','y1','y2', 'image_width', 'image_height', 'md5_hash', "average_hash"]] 		#[['frame','score','category', 'image_width', "image_height"]]

	# change frame from float to int
	tracking_matrix['frame'] = tracking_matrix['frame'].astype("int")

	# mock_1_tracking_matrix (currently frames)
	frames_tracking_matrix = tracking_matrix.drop_duplicates(subset=['frame']).sort(['frame'])['frame'].to_frame()

	# extract high score basketball and person from each frame
	single_pb_matrix = tracking_matrix.drop_duplicates(subset=['frame','category'], keep='first').sort(['score'], ascending=False)

	# group, rename, merge - exclude image_hight and width
	single_pb_category_matrix = single_pb_matrix.groupby('category')
	b_matrix = single_pb_category_matrix.get_group('basketball').drop(['category'], axis=1)
	p_matrix = single_pb_category_matrix.get_group('person').drop(['category'], axis=1)

	# merge basketball and person boxes with total frame range
	bpf_matrix = pd.merge(frames_tracking_matrix , pd.merge(b_matrix, p_matrix,on='frame', how='outer', suffixes=('_basketball', '_person')))

	# add radius to matrix
	bpfr_matrix = add_radii_column(bpf_matrix)

	#add iou to matrix
	bpfri_matrix = add_iou_column(bpfr_matrix)

	#add free bool to matrix
	bpfrif = add_free_column(bpfri_matrix)

	# add ball centerpoint columns
	print(add_basketball_box_center_column(bpfrif))

	#.rename(columns={'x1': 'bx1', 'x2': 'bx2', 'y1': 'by1', 'y2': 'by2'}, inplace=True)
	#.rename(columns={'x1': 'px1', 'x2': 'px2', 'y1': 'py1', 'p2': 'py2'}, inplace=True)
	"""
	#basketball_column_replacement_names = {'score': 'bscore', 'x1': 'bx1', 'x2': 'bx2', 'y1': 'by1', 'y2': 'by2'}
	#person_column_replacement_names = {'score': 'pscore', 'x1': 'px1', 'x2': 'px2', 'y1': 'py1', 'y2': 'py2'}
	#b_matrix.rename(columns=basketball_column_replacement_names, inplace=True)
	#p_matrix.rename(columns=person_column_replacement_names, inplace=True)

	"""



### testing

model_collection_name = "basketball_model_v1" 

shots = [2] #range(1,4)
"""
for shot_number in shots:

	MIN_SCORE = 1

	#input video frames directory path
	video_frames_dirpath = "/Users/ljbrown/Desktop/StatGeek/object_detection/video_frames/frames_shot_%s" % shot_number
	video_frames_dirpath = "/Users/ljbrown/Desktop/StatGeek/object_detection/video_frames/frames_shot_%s" % shot_number
	frame_info_bundel_filepath = "/Users/ljbrown/Desktop/StatGeek/object_detection/%s/image_evaluator_output/shot_%s_image_info_bundel.json" % (model_collection_name,shot_number)
	csv_video_frame_output_file_path = "/Users/ljbrown/Desktop/StatGeek/object_detection/%s/image_evaluator_output/shot_%s_frame_info_matrix.csv" % (model_collection_name,shot_number)

	#output csv
	output_csv_filepath = csv_video_frame_output_file_path


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




