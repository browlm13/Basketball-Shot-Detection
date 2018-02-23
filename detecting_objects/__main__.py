#!/usr/bin/env python

# internal
import logging
import os
import glob
import cv2
import json
import PIL.Image as Image
import numpy as np
import math

# my lib
#from image_evaluator.src import image_evaluator

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
	image_file_name = "frame_%d.JPEG" % frame 
	output_file = os.path.join(output_directory_path, image_file_name)
	cv2.imwrite(output_file, image_np)


#list of 4 coordanates for box
def draw_box_image_np(image_np, box):
	(left, right, top, bottom) = box
	cv2.rectangle(image_np,(left,top),(right,bottom),(0,255,0),3)
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


def get_high_score_box(image_info, category):
	category_box_score_tuple_list = get_category_box_score_tuple_list(image_info, category)

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





#center (x,y), color (r,g,b)
def draw_circle(image_np, center, radius=2, color=(0,0,255), thickness=10, lineType=8, shift=0):
	cv2.circle(image_np, center, radius, color, thickness=thickness, lineType=lineType, shift=shift)
	return image_np

def draw_person_ball_connector(image_np, person_mark, ball_mark):
	lineThickness = 7
	cv2.line(image_np, person_mark, ball_mark, (255,0,0), lineThickness)
	return image_np

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

"""
def write_frame_for_accuracy_test(output_directory_path, frame, image_np, image_info):

	image_file_name = "frame_%d.JPEG" % frame 

	for item in image_info:

		#test coors acuracy
		(left, right, top, bottom) = item['box']
		cv2.rectangle(image_np,(left,top),(right,bottom),(0,255,0),3)

	#write
	output_file = os.path.join(output_directory_path, image_file_name)
  cv2.imwrite(output_file, image_np)
 """

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


if __name__ == '__main__':

	#
	# Initial Evaluation
	#

	#video frames diretory (basketball_225.JPEG - basketball_262.JPEG)
	video_frames_dirpath = "/Users/ljbrown/Desktop/StatGeek/object_detection/video_frames/"

	#output images directory for checking
	output_image_directory = "/Users/ljbrown/Desktop/StatGeek/object_detection/output_images"

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

	#filter selected categories and min socre
	selected_categories_list = ['basketball', 'person']
	min_score_thresh = 1.0
	image_info_bundel = filter_selected_categories(filter_minimum_score_threshold(image_info_bundel, min_score_thresh), selected_categories_list)

	#get frame image paths in order
	frame_path_dict = get_frame_path_dict(video_frames_dirpath)

	#
	#	find distance between highest score player and highest score ball
	#

	#size of basketball : number of pixels
	#NBA is 29.5 inches in circumference
	#diameter = 29.5/(3.14) = 9.4 in

	#angle from person to basketball

	#person chest 3/4 height, 1/2 width
	#get person marker

	#
	# test load and write frame
	#

	ball_marks = []
	ball_radii = []

	person_marks = []


	for frame in range(225, 263):
		#frame = 259 #225
		frame_path = frame_path_dict[frame]
		image_np = load_image_np(frame_path)
		image_info = image_info_bundel[frame_path]

		person_box = get_high_score_box(image_info, 'person')
		person_mark = get_person_mark(person_box)

		ball_box = get_high_score_box(image_info, 'basketball')
		ball_mark = get_ball_mark(ball_box)
		ball_radius = get_ball_radius(ball_box)

		#add to history
		person_marks.append(person_mark)
		ball_marks.append(ball_mark)
		ball_radii.append(ball_radius)

		outside_mark = get_ball_outside_mark(person_box, ball_box)

		#test draw boxes
		#image_np = draw_all_boxes_image_np(image_np, image_info)
		#test draw basketball outline
		image_np = draw_circle(image_np, ball_mark, radius=ball_radius, color=(0,255,0), thickness=2)

		#draw history with decreasing brighness
		for i in range(len(ball_marks)):

			for j in range(i-1):
				k = j+1
				draw_person_ball_connector(image_np, ball_marks[j], ball_marks[k])

			image_np = draw_circle(image_np, ball_marks[i], color=(200,200,255))
			#image_np = draw_circle(image_np, person_marks[i], color=(200,200,255))

			#image_np = draw_circle(image_np, ball_marks[i], radius=ball_radii[i], color=(0,100,0), thickness=2)

		#test draw person mark
		image_np = draw_circle(image_np, person_mark)

		#test draw ball mark
		image_np = draw_circle(image_np, ball_mark)

		#test draw person ball connector
		#image_np = draw_person_ball_connector(image_np, person_mark, ball_mark)
		image_np = draw_person_ball_connector(image_np, person_mark, outside_mark)

		write_frame_for_accuracy_test(output_image_directory, frame, image_np)


	# test write video
	output_frame_paths_dict = get_frame_path_dict(output_image_directory)
	ordered_frame_paths = []
	for frame in range(225, 263):
		ordered_frame_paths.append(output_frame_paths_dict[frame])

	write_mp4_video(ordered_frame_paths, 'JPEG', 'output_video/tracking_3.mp4')
