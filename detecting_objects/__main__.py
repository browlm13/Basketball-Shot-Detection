#!/usr/bin/env python

import logging
import os
import glob
import cv2
import json
import PIL.Image as Image
import numpy as np
import math
import sys
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d							#	3d plotting
from scipy import stats 									# 	error rvalue/linear regression
from piecewise.regressor import piecewise 					# 	piecewise regression
from piecewise.plotter import plot_data_with_regression

# my lib
#from image_evaluator.src import image_evaluator

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#	note: #(left, right, top, bottom) = box

#
# Test accuracy by writing new images
#

#ext = extension
#source: http://tsaith.github.io/combine-images-into-a-video-with-python-3-and-opencv-3.html
def write_mp4_video(ordered_image_paths, ext, output_mp4_filepath):

	# Determine the width and height from the first image
	image_path = ordered_image_paths[0] 
	frame = cv2.imread(image_path)
	cv2.imshow('video',frame)
	height, width, channels = frame.shape

	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
	out = cv2.VideoWriter(output_mp4_filepath, fourcc, 20.0, (width, height))

	for image_path in ordered_image_paths:
		frame = cv2.imread(image_path)
		out.write(frame) # Write out frame to video

	# Release everything if job is finished
	out.release()

def write_frame_for_accuracy_test(output_directory_path, frame, image_np):
	# if image output directory does not exist, create it
	if not os.path.exists(output_directory_path): os.makedirs(output_directory_path)

	image_file_name = "frame_%d.JPEG" % frame 
	output_file = os.path.join(output_directory_path, image_file_name)
	
	#cv2.imwrite(output_file, image_np)	#BGR color

	#fix color
	image = Image.fromarray(image_np, 'RGB')
	image.save(output_file)
	logger.info("wrote %s to \n\t%s" % (image_file_name, output_file))

#list of 4 coordanates for box
def draw_box_image_np(image_np, box, color=(0,255,0)):
	(left, right, top, bottom) = box
	cv2.rectangle(image_np,(left,top),(right,bottom),color,3)
	return image_np

def draw_all_boxes_image_np(image_np, image_info):
	for item in image_info['image_items_list']:
		draw_box_image_np(image_np, item['box'])
	return image_np

def get_category_box_score_tuple_list(image_info, category):
	score_list = []
	box_list = []
	for item in image_info['image_items_list']:
		if item['class'] == category:
			box_list.append(item['box'])
			score_list.append(item['score'])
	return list(zip(score_list, box_list))

def get_high_score_box(image_info, category, must_detect=True):

	category_box_score_tuple_list = get_category_box_score_tuple_list(image_info, category)

	if len(category_box_score_tuple_list) == 0:
		logger.error("none detected: %s" % category)
		if must_detect:
			sys.exit()
			assert len(category_box_score_tuple_list) > 0
			high_score_index = 0
			high_score_value = 0

			index = 0
			for item in category_box_score_tuple_list:
				if item[0] > high_score_value:
					high_score_index = index
					high_score_value = item[0]
				index += 1

			return category_box_score_tuple_list[high_score_index][1]
		else:
			return None

	high_score_index = 0
	high_score_value = 0

	index = 0
	for item in category_box_score_tuple_list:
		if item[0] > high_score_value:
			high_score_index = index
			high_score_value = item[0]
		index += 1

	return category_box_score_tuple_list[high_score_index][1]


def get_person_mark(person_box):
	# 3/4 height, 1/2 width
	(left, right, top, bottom) = person_box
	width = int((right - left)/2)
	x = left + width
	height = int((bottom - top)*float(1.0/4.0))
	y = top + height
	return (x,y)

def get_ball_mark(ball_box):
	# 1/2 height, 1/2 width
	(left, right, top, bottom) = ball_box
	width = int((right - left)/2)
	x = left + width
	height = int((bottom - top)/2)
	y = top + height
	return (x,y)

def get_angle_between_points(mark1, mark2):
	x1, y1 = mark1
	x2, y2 = mark2
	radians = math.atan2(y1-y2,x1-x2)
	return radians

def get_ball_radius(ball_box):
	(left, right, top, bottom) = ball_box
	return int((right - left)/2)

def get_ball_outside_mark(person_box, ball_box):
	# mark on circumfrence of ball pointing twords person mark
	ball_mark = get_ball_mark(ball_box)
	person_mark = get_person_mark(person_box)

	ball_radius = get_ball_radius(ball_box)
	angle = get_angle_between_points(person_mark, ball_mark)

	dy = int(ball_radius * math.sin(angle))
	dx = int(ball_radius * math.cos(angle))

	outside_mark = (ball_mark[0] + dx, ball_mark[1] + dy)
	return outside_mark

#(left, right, top, bottom) = box
def box_area(box):
	(left, right, top, bottom) = box
	return (right-left) * (bottom-top)

def height_squared(box):
	(left, right, top, bottom) = box
	return (bottom-top)**2

#center (x,y), color (r,g,b)
def draw_circle(image_np, center, radius=2, color=(0,0,255), thickness=10, lineType=8, shift=0):
	cv2.circle(image_np, center, radius, color, thickness=thickness, lineType=lineType, shift=shift)
	return image_np

def draw_person_ball_connector(image_np, person_mark, ball_mark, color=(255,0,0)):
	lineThickness = 7
	cv2.line(image_np, person_mark, ball_mark, color, lineThickness)
	return image_np

def iou(box1, box2):
	#return "intersection over union" of two bounding boxes as float (0,1)
	paired_boxes = tuple(zip(box1, box2))

	# find intersecting box
	intersecting_box = (max(paired_boxes[0]), min(paired_boxes[1]), max(paired_boxes[2]), min(paired_boxes[3]))

	# adjust for min functions
	if (intersecting_box[1] < intersecting_box[0]) or (intersecting_box[3] < intersecting_box[2]):
		return 0.0

	# compute the intersection over union
	return box_area(intersecting_box)  / float( box_area(box1) + box_area(box2) - box_area(intersecting_box) )


def load_image_np(image_path):
	#non relitive path
	script_dir = os.path.dirname(os.path.abspath(__file__))
	image = Image.open(os.path.join(script_dir, image_path))
	(im_width, im_height) = image.size
	image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

	return image_np

def filter_minimum_score_threshold(image_info_bundel, min_score_thresh):
	filtered_image_info_bundel = {}
	for image_path, image_info in image_info_bundel.items():
		filtered_image_info_bundel[image_path] = image_info
		filtered_image_items_list = []
		for item in image_info['image_items_list']:
			if item['score'] > min_score_thresh:
				filtered_image_items_list.append(item)
		filtered_image_info_bundel[image_path]['image_items_list'] = filtered_image_items_list
	return filtered_image_info_bundel

def filter_selected_categories(image_info_bundel, selected_categories_list):
	filtered_image_info_bundel = {}
	for image_path, image_info in image_info_bundel.items():			
		filtered_image_info_bundel[image_path] = image_info
		filtered_image_items_list = []
		for item in image_info['image_items_list']:
			if item['class'] in selected_categories_list:
				filtered_image_items_list.append(item)
		filtered_image_info_bundel[image_path]['image_items_list'] = filtered_image_items_list
	return filtered_image_info_bundel		

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

def get_frame_path_dict(video_frames_dirpath):

	# get path to each frame in video frames directory
	image_path_list = glob.glob(video_frames_dirpath + "/*")

	frame_path_dict = []
	for path in image_path_list:

		#get filename
		filename = os.path.basename(path)

		#strip extension
		filename_wout_ext = filename.split('.')[0]
		
		#frame_number
		frame = int(filename_wout_ext.split('_')[1])

		frame_path_dict.append((frame, path))

	return dict(frame_path_dict)

# return minimum and maximum frame number for frame path dict as well as continous boolean value
def min_max_frames(frame_path_dict):
	frames, paths = zip(*frame_path_dict.items())

	min_frame, max_frame = min(frames), max(frames)
	continuous = set(range(min_frame,max_frame+1)) == set(frames)

	return min_frame, max_frame, continuous 

def frame_directory_to_video(input_frames_directory, output_video_file):

	# write video
	output_frame_paths_dict = get_frame_path_dict(input_frames_directory)
	min_frame, max_frame, continuous = min_max_frames(output_frame_paths_dict)

	if continuous:
		ordered_frame_paths = []
		for frame in range(min_frame, max_frame + 1):
			ordered_frame_paths.append(output_frame_paths_dict[frame])
		write_mp4_video(ordered_frame_paths, 'JPEG', output_video_file)
	else:
		logger.error("Video Frames Directory %s Not continuous")

#
# pure boundary box image (highscore person and ball in image info)
#
def pure_boundary_box_frame(frame_image, image_info):

	#load a frame for size and create black image
	rgb_blank_image = np.zeros(frame_image.shape)

	#get person and ball boxes
	person_box = get_high_score_box(image_info, 'person', must_detect=False)
	ball_box = get_high_score_box(image_info, 'basketball', must_detect=False)

	# draw boxes (filled)
	if person_box is not None:
		(left, right, top, bottom) = person_box
		cv2.rectangle( rgb_blank_image, (left, top), (right, bottom), color=(255,50,50), thickness=-1, lineType=8 )

	if ball_box is not None:
		(left, right, top, bottom) = ball_box
		cv2.rectangle( rgb_blank_image, (left, top), (right, bottom), color=(30,144,255), thickness=-1, lineType=8 )

	return rgb_blank_image

#
# stabalize to person mark, scale to ball box (highscore person and ball in image info)
#
def stabalize_to_person_mark_frame(frame_image, image_info):

	#load a frame for size and create black image
	rgb_blank_image = np.zeros(frame_image.shape)
	rgb_blank_image = frame_image

	#get person and ball boxes
	person_box = get_high_score_box(image_info, 'person', must_detect=False)
	ball_box = get_high_score_box(image_info, 'basketball', must_detect=False)

	if person_box is not None:
		#use person mark as center coordinates
		px, py = get_person_mark(person_box)

		height, width, depth = rgb_blank_image.shape
		center = (int(width/2), int(height/2))

		# draw person box
		person_left, person_right, person_top, person_bottom = person_box
		person_width = person_right - person_left
		person_height = person_bottom - person_top

		new_person_left = center[0] - int(person_width/2)
		new_person_right = center[0] + int(person_width/2)
		new_person_top = center[1] - int(person_height * (1/4))
		new_person_bottom = center[1] + int(person_height * (3/4))

		new_person_box = (new_person_left, new_person_right, new_person_top,new_person_bottom)


		if ball_box is not None:

			#use person mark as center coordinates
			bx, by = get_ball_mark(ball_box)

			height, width, depth = rgb_blank_image.shape
			center = (int(width/2), int(height/2))

			new_bx = bx - px + center[0]
			new_by = by - py + center[1]
			new_ball_mark = (new_bx, new_by) 

			ball_radius = get_ball_radius(ball_box)

			#draw_circle(rgb_blank_image, new_ball_mark)
			#draw_circle(rgb_blank_image, new_ball_mark, radius=ball_radius)

			#old  drawing
			draw_box_image_np(rgb_blank_image, person_box)
			draw_circle(rgb_blank_image, (px, py))
			draw_box_image_np(rgb_blank_image, ball_box) #ball box
			draw_circle(rgb_blank_image, (bx, by))	#ball circle
			draw_person_ball_connector(rgb_blank_image, (px,py), (bx,by)) #draw connectors


			#iou overlap
			if iou(person_box, ball_box) > 0:

				#new coordinate drawings

				#ball
				draw_circle(rgb_blank_image, new_ball_mark, color=(0,255,0))	#mark
				draw_circle(rgb_blank_image, new_ball_mark, radius=ball_radius, color=(0,255,0), thickness=5) #draw ball
				draw_person_ball_connector(rgb_blank_image, center, new_ball_mark, color=(0,255,0)) # connector

				#person
				draw_circle(rgb_blank_image, center, color=(0,255,0))
				draw_box_image_np(rgb_blank_image, new_person_box, color=(0,255,0))

			else:

				#new coordinate drawings

				#ball
				draw_circle(rgb_blank_image, new_ball_mark, color=(0,0,255))	#mark
				draw_circle(rgb_blank_image, new_ball_mark, radius=ball_radius, color=(0,0,255)) #ball
				draw_person_ball_connector(rgb_blank_image, center, new_ball_mark, color=(0,0,255)) #connector

				#person
				draw_circle(rgb_blank_image, center, color=(0,0,255))
				draw_box_image_np(rgb_blank_image, new_person_box, color=(0,0,255))

	return rgb_blank_image


# run frame cycle and execute function at each step passing current frame path to function, and possibly more
# cycle function should return image after each run
# output frame_path_dict should be equivalent except to output directory
def frame_cycle(image_info_bundel, input_frame_path_dict, output_frames_directory, output_video_file, cycle_function, apply_history=False):
	# get minimum and maximum frame indexes
	min_frame, max_frame, continuous = min_max_frames(input_frame_path_dict)

	# frame cycle
	if continuous:
	
		for frame in range(min_frame, max_frame + 1):
			frame_path = input_frame_path_dict[frame]
			image_info = image_info_bundel[frame_path]

			frame_image = cv2.imread(frame_path)	#read image
			image_np = cycle_function(frame_image, image_info)

			if apply_history:

				# TODO: fix weights
				for i in range(frame, min_frame, -1):
					alpha = 0.1
					beta = 0.1
					gamma = 0.5
					i_frame_path = frame_path_dict[i]
					i_image_info = image_info_bundel[i_frame_path]
					i_frame_image = cv2.imread(i_frame_path)	#read image
					next_image_np = cycle_function(i_frame_image, i_image_info)
					image_np = cv2.addWeighted(image_np,alpha,next_image_np,beta,gamma)

			# write images
			write_frame_for_accuracy_test(output_frames_directory, frame, image_np)

		# write video
		frame_directory_to_video(output_frames_directory, output_video_file)
	else:
		logger.error("not continuous")

# source: https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy
def group_consecutives(vals, step=1):
	"""Return list of consecutive lists of numbers from vals (number list)."""
	run = []
	result = [run]
	expect = None
	for v in vals:
		if (v == expect) or (expect is None):
			run.append(v)
		else:
			run = [v]
			result.append(run)
		expect = v + step
	return result

def group_consecutives_by_column(matrix, column, step=1):
	vals = matrix[:,column]
	runs = group_consecutives(vals, step)

	run_range_indices = []
	for run in runs:
		start = np.argwhere(matrix[:,column] == run[0])[0,0]
		stop = np.argwhere(matrix[:,column] == run[-1])[0,0] + 1
		run_range_indices.append([start,stop])

	#split matrix into segments (smaller matrices)
	split_matrices = []
	for run_range in run_range_indices:
		start, stop = run_range
		trajectory_matrix = matrix[start:stop,:]
		split_matrices.append(trajectory_matrix)
	
	return split_matrices

#
# 							ball collected data points matrix (ball_cdpm)
#
# columns: frame, x, y, ball state / iou bool
#
# enum keys: 
#			'frame column', 'ball mark x column', 'ball mark y column', 'ball state column'		#columns 
#			'no data', 'free ball', 'held ball'													#ball states
#
# "essentially an alternative representation of image_info_bundel"
# to access: frames, xs, ys, state = ball_cdpm.T
def get_ball_cdpm( ball_cdpm_enum, input_frame_path_dict, image_info_bundel):

		# get minimum and maximum frame indexes
		min_frame, max_frame, continuous = min_max_frames(input_frame_path_dict)

		#	Matrix - fill with no data

		num_rows = (max_frame + 1) - min_frame
		num_cols = 4						# frame, ballmark x, ballmark y, ball state (iou bool)
		ball_cdpm = np.full((num_rows, num_cols), ball_cdpm_enum['no data'])

		# iou boolean lambda function for 'ball mark x column'
		get_ball_state = lambda person_box, ball_box : ball_cdpm_enum['held ball'] if (iou(person_box, ball_box) > 0) else ball_cdpm_enum['free ball']

		# 					Fill ball collected data points matrix (ball_cdpm)
		#
		# 					'frame', 'ballmark x', 'ballmark y', 'ball state'

		index = 0
		for frame in range(min_frame, max_frame + 1):
			frame_path = input_frame_path_dict[frame]
			frame_info = image_info_bundel[frame_path]

			#get frame ball box and frame person box
			frame_ball_box = get_high_score_box(frame_info, 'basketball', must_detect=False)
			frame_person_box = get_high_score_box(frame_info, 'person', must_detect=False)

			# frame number column 'frame column'
			ball_cdpm[index,ball_cdpm_enum['frame column']] = frame

			#ball mark column 'ball mark x column', 'ball mark y column' (x,y)
			if (frame_ball_box is not None):
				frame_ball_mark = get_ball_mark(frame_ball_box)
				ball_cdpm[index,ball_cdpm_enum['ball mark x column']] = frame_ball_mark[0]
				ball_cdpm[index,ball_cdpm_enum['ball mark y column']] = frame_ball_mark[1]

			#ball state/iou bool column 'ball state column ''
			if (frame_ball_box is not None) and (frame_person_box is not None):
				ball_cdpm[index,ball_cdpm_enum['ball state column']] = get_ball_state(frame_person_box, frame_ball_box)

			index += 1

		# return matrix
		return ball_cdpm

#
#
#											Main
#
#

if __name__ == '__main__':

	#
	# Initial Evaluation
	#

	for i in range(16,17):

		print ("video %d" % i)

		model_collection_name = "basketball_model_v1" #"person_basketball_model_v1"

		#input video frames directory paths
		video_frames_dirpath = "/Users/ljbrown/Desktop/StatGeek/object_detection/video_frames/frames_shot_%s" % i

		#output images and video directories for checking
		output_frames_directory = "/Users/ljbrown/Desktop/StatGeek/object_detection/%s/output_images/output_frames_shot_%s" % (model_collection_name,i)
		output_video_file = '%s/output_video/shot_%d_prediction_trajectory.mp4' % (model_collection_name,i)

		#image_boolean_bundel and image_info_bundel file paths for quick access
		image_boolean_bundel_filepath = "/Users/ljbrown/Desktop/StatGeek/object_detection/%s/image_evaluator_output/shot_%s_image_boolean_bundel.json" % (model_collection_name,i)
		image_info_bundel_filepath = "/Users/ljbrown/Desktop/StatGeek/object_detection/%s/image_evaluator_output/shot_%s_image_info_bundel.json" % (model_collection_name,i)

		#create dirs*** if dont exist: image_info_bundel_filepath, image_boolean_bundel_filepath,  

		#tensorflow models
		BASKETBALL_MODEL = {'name' : 'basketball_model_v1', 'use_display_name' : False, 'paths' : {'frozen graph': "image_evaluator/models/basketball_model_v1/frozen_inference_graph/frozen_inference_graph.pb", 'labels' : "image_evaluator/models/basketball_model_v1/label_map.pbtxt"}}
		PERSON_MODEL = {'name' : 'ssd_mobilenet_v1_coco_2017_11_17', 'use_display_name' : True, 'paths' : {'frozen graph': "image_evaluator/models/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph/frozen_inference_graph.pb", 'labels' : "image_evaluator/models/ssd_mobilenet_v1_coco_2017_11_17/mscoco_label_map.pbtxt"}}
		BASKETBALL_PERSON_MODEL = {'name' : 'person_basketball_model_v1', 'use_display_name' : False, 'paths' : {'frozen graph': "image_evaluator/models/person_basketball_model_v1/frozen_inference_graph/frozen_inference_graph.pb", 'labels' : "image_evaluator/models/person_basketball_model_v1/label_map.pbtxt"}}
		#bool rule - any basketball or person above an accuracy score of 40.0
		bool_rule = "any('basketball', 40.0) or any('person', 40.0)"

		#
		# 	  evaluate frame directory
		#				and
		#	save to files for quick access
		#

		#save_image_directory_evaluations(video_frames_dirpath, image_boolean_bundel_filepath, image_info_bundel_filepath, [BASKETBALL_MODEL, PERSON_MODEL], bool_rule)
		#save_image_directory_evaluations(video_frames_dirpath, image_boolean_bundel_filepath, image_info_bundel_filepath, [BASKETBALL_PERSON_MODEL], bool_rule)

		#
		#	load previously evaluated frames
		#

		#load saved image_info_bundel
		image_info_bundel = load_image_info_bundel(image_info_bundel_filepath)

		#filter selected categories and min socre
		selected_categories_list = ['basketball', 'person']
		min_score_thresh = 10.0
		image_info_bundel = filter_selected_categories(filter_minimum_score_threshold(image_info_bundel, min_score_thresh), selected_categories_list)

		#get frame image paths in order
		input_frame_path_dict = get_frame_path_dict(video_frames_dirpath)


		#
		#	Call function for frame cycle
		#

		#output_video_file = 'output_video/shot_%d_pure_block_history.mp4' % i
		#frame_cycle(image_info_bundel, input_frame_path_dict, output_frames_directory, output_video_file, pure_boundary_box_frame, apply_history=True)
		#frame_cycle(image_info_bundel, input_frame_path_dict, output_frames_directory, output_video_file, stabalize_to_person_mark_frame)


		#
		# 	Find Ball Trajectory Formula
		#	Mock 1: Assertions: Stable video, 1 person, 1 ball

		#
		#	Build ball data points matrix  (ball_cdpm)
		#													columns: frame, ball mark x, ball mark y, ball state
		#													ball states: no data, held ball, free ball

		ball_cdpm_enum = {
			'no data' : -1,
			'free ball' : 1,
			'held ball' : 0,
			'frame column' : 0,
			'ball mark x column' : 1,
			'ball mark y column' : 2,
			'ball state column' : 3
		}

		ball_cdpm = get_ball_cdpm(ball_cdpm_enum, input_frame_path_dict, image_info_bundel)


		#
		#	Plot ball data points matrix  (ball_cdpm)
		#
		#		- 2d or 3d
		#		- colored on ball state (held/free ... no data points are dropped)
		#

		# std error x value linear regression threshold to apply piecewise linear regression
		STD_ERROR_THRESHOLD = 0.5

		# Write frames
		WRITE_FRAMES = False

		# Write Video
		WRITE_VIDEO = False

		# display plot or show regression segmentation, NOT both
		SHOW_PLOT = False
		SHOW_REGRESSION_SEGMENTATION = True
		SHOW_REGRESSION_SEGMENTATION_Y = False # default show x calculation

		# plotting in 3D
		PLOT_3D = False

		# cut off regression formula trajectories (set minimum and maximum x and y values from min and maxs of actual datapoints observed)
		TRIM_PLOT = False

		# flip y values to compensate for wierdo image y coordinates
		INVERT_Y_VALUES = True

		# ball state enumn 
		ball_state_enum = {
			'no data' : -1,
			'free ball' : 1,
			'held ball' : 0
		}

		# color/ball state map
		ball_state_colors_enum = {
			'free ball' : 'g',
			'held ball' : 'r'
		}

		#remove no data elements
		cleaned_ball_cdpm = ball_cdpm[ball_cdpm[:, ball_cdpm_enum['ball mark x column']] != ball_cdpm_enum['no data'], :] 	# extract all rows where there is data 
		frames, xs, ys, ball_states = cleaned_ball_cdpm.T

		# invert y values
		if INVERT_Y_VALUES:
			neg = lambda t: t*(-1)
			invert_array = np.vectorize(neg)
			ys = invert_array(ys)

		# change color for ball state
		inv_ball_state_enum = { v: k for k, v in ball_state_enum.items()}	# inverse map of ball state enum
		ball_state_colors = [] 
		for bs in ball_states:
			i_state = inv_ball_state_enum[bs]
			color = ball_state_colors_enum[i_state]
			ball_state_colors.append(color)
	
		if SHOW_PLOT:
			assert not SHOW_REGRESSION_SEGMENTATION
			# add plots
			fig = plt.figure()
			if PLOT_3D == True:
				ax = plt.axes(projection='3d')
				ax.scatter3D(xs, ys, frames, c=ball_state_colors, cmap='Greens')
				ax.set_xlabel('Xs')
				ax.set_ylabel('Ys')
				ax.set_zlabel('frames')
			else:
				for i in range(len(xs)):
					plt.scatter(xs[i], ys[i], color=ball_state_colors[i])
					plt.xlabel('Xs', fontsize=18)
					plt.ylabel('Ys', fontsize=18)
		
		
		#
		# Break ball data points matrix into multiple submatrices (free_ball_cbdm's) where ball is free
		#				(break around held ball datapoints)
		#

		# cut ball_cdpm at frames with ball state column value == 'held ball', leaving only free ball datapoints in an array of matrices
		free_ball_cbdm_array = ball_cdpm[ball_cdpm[:, ball_cdpm_enum['ball state column']] != ball_cdpm_enum['held ball'], :] 	# extract all rows with the ball state column does not equal held ball
		free_ball_cbdm_array = group_consecutives_by_column(free_ball_cbdm_array, ball_cdpm_enum['frame column'])						# split into seperate matrices for ranges

		# 
		#	find regression formula (ball trajectory formula) for each free_ball_cbdm (sub matrix)
		#

		shot_regression_formulas = []	# [[regression_shot_xs, regression_shot_ys, shot_frames], ...]

		for i in range(len(free_ball_cbdm_array)):
			possible_trajectory_coordinates = free_ball_cbdm_array[i]		# 'possible' ball trajectory coordinates, ranges without held ball states tagged by model

			# extract 'known' ball trajectory coordinates, tagged by model
			#remove missing datapoints 
			known_trajectory_points = possible_trajectory_coordinates[possible_trajectory_coordinates[:, ball_cdpm_enum['ball mark x column']] != ball_cdpm_enum['no data'], :] 	# extract all rows where there is data 
			kframes, kxs, kys, kstate = known_trajectory_points.T

			#enure known ball trajectory has more than 1 data point
			if len(kframes) > 1:

				#total frame range for plotting regresion lines - so path is not line segments
				trajectory_total_frame_range = np.linspace(kframes[0], kframes[-1], kframes[-1] - kframes[0])

				#
				#						peicewise linear regression
				#
				# 	Apply if x std error is abole threshold
				#	Apply peicewise linear regression to x values.
				#	x values should change linearly if ball is in free flight (ignoring air resistance).
				#	Use peicewise linear regression to find the point at which the free flying ball hits another object/changes its path
				#	Find point of intersection for seperate regression lines to find final frame of shot ball trajectory
				#

				# test linear regression std error for x values
				slope, intercept, r_value, p_value, std_err = stats.linregress(kframes, kxs)
				start_frame, stop_frame = kframes[0], kframes[-1]
				shot_frame_range = [start_frame, stop_frame]
				shot_frames = np.linspace(start_frame, stop_frame, stop_frame-start_frame)

				# regression (not piecewise)
				#xs - degreen 1 regression fit (cleaned xs - cxs)
				p1 = np.polyfit(kframes, kxs, 1)
				regression_shot_xs = np.polyval(p1, shot_frames)

				# apply peicewise linear regression only if x std error is above threshold
				if std_err >= STD_ERROR_THRESHOLD:

					#peicewise model
					model = piecewise(kframes, kxs)
					start_frame, stop_frame, coeffs = model.segments[0]		# find start and stopping frame (start_frame, stop_frame) from first line segment, and line formula
					shot_frame_range = [start_frame, stop_frame]

					shot_frames = np.linspace(start_frame, stop_frame, stop_frame-start_frame)	# final
					regression_shot_xs = np.polyval(list(coeffs)[::-1], shot_frames)			# final


				#
				# Find ys - degree 2 regression fit capped at shot frame range above
				#

				# remove y frames from trajectory not covered by shot frame
				cap_index = np.where(kframes==stop_frame)[0][0]
				capped_ys = kys[:cap_index]
				capped_xs = kxs[:cap_index]
				capped_frames = kframes[:cap_index]


				# invert y values
				if INVERT_Y_VALUES:
					neg = lambda t: t*(-1)
					invert_array = np.vectorize(neg)
					capped_ys = invert_array(capped_ys)

				# find y values degree 2 regression fit
				p2 = np.polyfit(capped_frames, capped_ys, 2)
				regression_shot_ys = np.polyval(p2, shot_frames)							# final
			
				if SHOW_PLOT:
					assert not SHOW_REGRESSION_SEGMENTATION
					# Add to plot
					if PLOT_3D == True:
						ax.plot3D(regression_shot_xs, regression_shot_ys, shot_frames, c=np.random.rand(3,1), linewidth=1)
					else:
						plt.plot(regression_shot_xs, regression_shot_ys, c=np.random.rand(3,1), markersize=1)

				# Add to shot_regression_formulas
				shot_regression_formulas.append([regression_shot_xs, regression_shot_ys, shot_frames])

				# invert regression y values
				if INVERT_Y_VALUES:
					neg = lambda t: t*(-1)
					invert_array = np.vectorize(neg)
					regression_shot_ys = invert_array(regression_shot_ys)

				if SHOW_REGRESSION_SEGMENTATION:
					assert not SHOW_PLOT
					if SHOW_REGRESSION_SEGMENTATION_Y:
						#plt.scatter(capped_frames, capped_ys)			
						plt.plot(shot_frames, regression_shot_ys)
						plt.scatter(kframes, kys)
					else:
						#plt.scatter(capped_frames, capped_xs)
						plt.plot(shot_frames, regression_shot_xs)
						plt.scatter(kframes, kxs)
				#plt.scatter(capped_frames, capped_ys)			
				#plt.plot(shot_frames, regression_shot_ys)
				#plt.scatter(kframes, kys)

		# set minimum and maximum x and y values from min and maxs of actual datapoints observed
		# cut off regression formula trajectories
		if TRIM_PLOT:
			axes = plt.gca()
			axes.set_xlim([xs.min(),xs.max()])
			axes.set_ylim([ys.min(),ys.max()])

		#display
		plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

		if SHOW_PLOT or SHOW_REGRESSION_SEGMENTATION:
			plt.show()

		#if SAVE_PLOT:
		#fig.savefig('samplefigure', bbox_inches='tight')

		# drawing predicted lines onto actual video frames from 2d plots
		# get minimum and maximum frame indexes

		if WRITE_FRAMES or WRITE_VIDEO:
			min_frame, max_frame, continuous = min_max_frames(input_frame_path_dict)
			assert continuous
			for frame in range(min_frame,max_frame+1) :

				#load
				frame_path = input_frame_path_dict[frame]
				frame_image = cv2.imread(frame_path)	#read image
				frame_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)

				for srf in shot_regression_formulas:
					#draw
					srf_xs, srf_ys, srf_frames = srf
					points = list(zip(srf_xs, srf_ys))
					cv2.polylines(frame_image,  np.int32([points]), False, (0,255,0), thickness=5) 

				#display
				#frame_image = Image.fromarray(frame_image, 'RGB')
				#frame_image.show()
				#frame_image.save(PATHTOSAVE)

				# write frames
				write_frame_for_accuracy_test(output_frames_directory, frame, frame_image)

		# write video
		if WRITE_VIDEO:
			frame_directory_to_video(output_frames_directory, output_video_file)


