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


import sys
import cv2
#import numpy as np
from matplotlib import pyplot as plt

# my lib
#from image_evaluator.src import image_evaluator

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#note: #(left, right, top, bottom) = box

dodgerblue = (30,144,255)

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
	cv2.imwrite(output_file, image_np)


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


# return minimum and maximum frame number for frame path dict as well as continous boolean value
def min_max_frames(frame_path_dict):
	frames, paths = zip(*frame_path_dict.items())

	min_frame, max_frame = min(frames), max(frames)
	continuous = set(range(min_frame,max_frame+1)) == set(frames)

	return min_frame, max_frame, continuous 


"""
def get_output_frame_path_dict(input_frame_path_dict, output_frames_directory):

	output_frame_path_dict = {}
	for frame, path in input_frame_path_dict.items():
		new_path = os.path.join(output_frames_directory, os.path.basename(path))
		output_frame_path_dict[frame] = new_path
	return output_frame_path_dict
"""

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
#	apply fuction for image info history (up until frame of specified attribute)
#

# use addWeighted instead
"""
#def get_history(image_info):
def apply_function_to_history(frame_function, frame, input_frame_path_dict, image_info_bundel, fade=False):
	# get minimum and maximum frame indexes
	min_frame, max_frame, continuous = min_max_frames(input_frame_path_dict)

	# apply to each frame starting from first
	if continuous:
		frame_path = input_frame_path_dict[min_frame]
		image_np = cv2.imread(frame_path)	#read image
		frame_path = input_frame_path_dict[frame]
		image_info = image_info_bundel[frame_path]
		image_np = frame_function(image_np, image_info, blackout=True)
		for frame in range(min_frame+1, frame+1):

			frame_path = input_frame_path_dict[frame]
			image_info = image_info_bundel[frame_path]
			image_np = frame_function(image_np, image_info, blackout=False)

		return image_np

	else:
		logger.error("not continuous")
"""

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
		global dodgerblue
		cv2.rectangle( rgb_blank_image, (left, top), (right, bottom), color=(255,50,50), thickness=-1, lineType=8 )

	if ball_box is not None:
		(left, right, top, bottom) = ball_box
		cv2.rectangle( rgb_blank_image, (left, top), (right, bottom), color=dodgerblue, thickness=-1, lineType=8 )

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
	min_frame, max_frame, continuous = min_max_frames(frame_path_dict)

	# frame cycle
	if continuous:
	
		for frame in range(min_frame, max_frame + 1):
			frame_path = frame_path_dict[frame]
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


if __name__ == '__main__':

	#
	# Initial Evaluation
	#

	for i in range(1,18):

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

		#save to files for quick access
		#save_image_directory_evaluations(video_frames_dirpath, image_boolean_bundel_filepath, image_info_bundel_filepath, [BASKETBALL_MODEL, PERSON_MODEL], bool_rule)
		#save_image_directory_evaluations(video_frames_dirpath, image_boolean_bundel_filepath, image_info_bundel_filepath, [BASKETBALL_PERSON_MODEL], bool_rule)


		#load saved image_info_bundel
		image_info_bundel = load_image_info_bundel(image_info_bundel_filepath)

		#filter selected categories and min socre
		selected_categories_list = ['basketball', 'person']
		min_score_thresh = 10.0
		image_info_bundel = filter_selected_categories(filter_minimum_score_threshold(image_info_bundel, min_score_thresh), selected_categories_list)

		#get frame image paths in order
		frame_path_dict = get_frame_path_dict(video_frames_dirpath)



		#
		#	Call function for frame cycle
		#

		input_frame_path_dict = get_frame_path_dict(video_frames_dirpath)
		#output_video_file = 'output_video/shot_%d_pure_block_history.mp4' % i
		#frame_cycle(image_info_bundel, input_frame_path_dict, output_frames_directory, output_video_file, pure_boundary_box_frame, apply_history=True)
		#frame_cycle(image_info_bundel, input_frame_path_dict, output_frames_directory, output_video_file, stabalize_to_person_mark_frame)



		#
		# Find ball trajectory
		#
		# get minimum and maximum frame indexes
		min_frame, max_frame, continuous = min_max_frames(frame_path_dict)

		# frame cycle
		if continuous:
		
			#
			#histories for ball trajectory plotting
			#
			person_marks = {}
			ball_marks = {}

			stop_collecting = False
			for frame in range(min_frame, max_frame + 1):
				frame_path = frame_path_dict[frame]
				image_info = image_info_bundel[frame_path]

				#get person and ball boxes
				person_box = get_high_score_box(image_info, 'person', must_detect=False)
				ball_box = get_high_score_box(image_info, 'basketball', must_detect=False)

				if person_box is not None:

						if not stop_collecting:
							#histories for ball trajectory plotting
							person_marks[frame] = get_person_mark(person_box)

						if ball_box is not None:

							#iou overlap
							if (iou(person_box, ball_box) == 0) and (not stop_collecting):
								#histories for ball trajectory plotting
								ball_marks[frame] = get_ball_mark(ball_box)
								print(frame)

							#break on rejoin
							if (iou(person_box, ball_box) > 0) and (bool(ball_marks)):
								stop_collecting = True
			



			"""
				starting at first frame ball is detected
				make create element for each possible frame from detection to end
				fill each known gap with balls shifted coordinates
					shifted coordinates: bx-px, by-py
			"""

			known_shifted_coordinates = []
			frame_number = []

			shifted_ball_marks = {}
			shifted_ball_marks_x = {}
			shifted_ball_marks_y = {}

			for frame, ball_mark in ball_marks.items():
				person_mark = person_marks[frame]
				new_ball_mark = (ball_mark[0] - person_mark[0], ball_mark[1] - person_mark[1])
				shifted_ball_marks[frame] = new_ball_mark

				shifted_ball_marks_x[frame] = ball_mark[0] - person_mark[0]
				#shifted_ball_marks_y[frame] = person_mark[1]- ball_mark[1] #ball_mark[1] - person_mark[1]
				shifted_ball_marks_y[frame] = ball_mark[1] - person_mark[1]


			frames, shifted_xs = zip(*shifted_ball_marks_x.items())
			frames, shifted_ys = zip(*shifted_ball_marks_y.items())

			#print(frames)
			#print(shifted_xs)
			#print(shifted_ys)

			np_frames = np.array(list(frames))
			np_shifted_xs = np.array(list(shifted_xs))
			np_shifted_ys = np.array(list(shifted_ys))

			fig = plt.figure()
			#ax = fig.add_subplot(1, 1, 1)
			#ax.scatter(np_frames, shifted_xs)
			#plt.show()

			#find regression line for x versus frames
			#from scipy import stats
			#slope, intercept, r_value, p_value, std_err = stats.linregress(np_frames, shifted_xs)
			#print("slope: %f" % slope)
			#print("intercept: %f" % intercept)
			#print("r_value: %f" % r_value)

			#xs - degreen 1 regression fit
			p1 = np.polyfit(np_frames, shifted_xs, 1)
			#plt.plot(np_frames, np.polyval(p1,np_frames))

			#ys - degree 2 regression fit
			p2 = np.polyfit(np_frames, shifted_ys, 2)
			#plt.plot(np_frames, np.polyval(p2,np_frames), 'r-')
			#plt.show()


			# use new regression fit formulas to find actual values at x

			regression_shifted_ball_marks = []

			for frame in range(np_frames.min(), np_frames.max() +1):
				shifted_x_ball_mark = np.polyval(p1, frame)
				shifted_y_ball_mark = np.polyval(p2, frame)
				regression_shifted_ball_marks.append((shifted_x_ball_mark,shifted_y_ball_mark))

			regression_shifted_ball_marks_x, regression_shifted_ball_marks_y = zip(*regression_shifted_ball_marks)
			#plt.plot(regression_shifted_ball_marks_x, regression_shifted_ball_marks_y, 'r')
			#plt.plot(regression_shifted_ball_marks_y, regression_shifted_ball_marks_x, 'r')
			#plt.plot(regression_shifted_ball_marks_y, np_frames)
			#plt.show()


			"""
			# convert regression shifted ballmarks to original axis
			regression_ball_marks = {}
			for frame in range(np_frames.min(), np_frames.max() +1):
				shifted_x_ball_mark = np.polyval(p1, frame)
				shifted_y_ball_mark = np.polyval(p2, frame)

				person_mark = person_marks[frame]

				#old_axis_ball_mark = (shifted_x_ball_mark + person_mark[0], person_mark[1] + shifted_y_ball_mark)
				old_axis_ball_mark = (int(shifted_x_ball_mark + person_mark[0]), int(shifted_y_ball_mark + person_mark[1]))
				regression_ball_marks[frame] = old_axis_ball_mark

			print(regression_ball_marks)
			print(ball_marks)



			#write to images / video
			for frame in range(min_frame, max_frame + 1):
				frame_path = frame_path_dict[frame]
				frame_image = cv2.imread(frame_path)	#read image

				if frame == np_frames.min():
					draw_circle(frame_image, regression_ball_marks[frame], color=(0,0,255))	#mark

				# write images
				write_frame_for_accuracy_test(output_frames_directory, frame, frame_image)

			# write video
			frame_directory_to_video(output_frames_directory, output_video_file)

			"""
			#only collect for acceptable range

			regression_ball_marks = []
			prev_person_mark = person_marks[np_frames.min()]
			for frame in range(np_frames.min(), np_frames.max() +1):
				shifted_x_ball_mark = np.polyval(p1, frame)
				shifted_y_ball_mark = np.polyval(p2, frame)

				person_mark = person_marks[frame]

				if (person_mark is not None):
					#old_axis_ball_mark = (shifted_x_ball_mark + person_mark[0], person_mark[1] + shifted_y_ball_mark)
					old_axis_ball_mark = (int(shifted_x_ball_mark + person_mark[0]), int(shifted_y_ball_mark + person_mark[1]))
					regression_ball_marks.append(old_axis_ball_mark)

					prev_person_mark = person_mark
				else:
					old_axis_ball_mark = (int(shifted_x_ball_mark + prev_person_mark[0]), int(shifted_y_ball_mark + prev_person_mark[1]))
					regression_ball_marks.append(old_axis_ball_mark)

			#print(regression_ball_marks)
			#print(ball_marks)


			#write to images / video
			count = 0
			for frame in range(min_frame, max_frame + 1):

				try: 
					frame_path = frame_path_dict[frame]
					frame_image = cv2.imread(frame_path)	#read image

					if (frame >= np_frames.min()) and (frame <= np_frames.max()): #<
						draw_circle(frame_image, regression_ball_marks[count], color=(0,0,255))	#mark
						count = count +1

					# write images
					write_frame_for_accuracy_test(output_frames_directory, frame, frame_image)

				except: pass

			# write video
			frame_directory_to_video(output_frames_directory, output_video_file)

			"""

			#ax.scatter(np_frames, shifted_ys)
			#plt.show()

			#find regression line for x versus frames
			#from scipy import stats
			#slope, intercept, r_value, p_value, std_err = stats.linregress(np_frames, shifted_xs)
			#print("slope: %f" % slope)
			#print("intercept: %f" % intercept)
			#print("r_value: %f" % r_value)


			#not quite the mean
			#print(np.mean(np_shifted_xs))

			#find the mean
			#frame_difrence = np.maximum(np_frames) - np.minimum(np_frames)
			"""



			"""
			for frame in range(min_frame, max_frame + 1):
				if frame in ball_marks:
					frame_number.append()
			"""

	


		"""
			-ball should move equal amounts horizontally each frame after shot
			-matrix of ball mark center points -> evenly space points off of mean seperation for path estimation
			-you can aply gravitys pull and calculate the mean seperation for veriticle points also
			-try applying these rules to centering around person mark vs original

			1. find array of person marks
			2. find array of ball marks
			3. calculate distance to ball where person is origin array of ball marks (shifted ball marks)
			4. plot
			5. find mean x seperation of shifted ball marks
			6. linspace x ball marks according to that

			gravity correction
			(assume like 24 frames a second)
			it will be a parabola that explains the speed or distnace downward or upward
		"""
		"""
		#plt.imshow(rgb_blank_image)
		#plt.show()

		#
		# test load and write frame
		

		ball_marks = []
		ball_radii = []
		person_marks = []

		# stats
		person_boxes = []
		ball_boxes = []

		# get frames range
		min_frame = 1
		max_frame = 1
		for key in frame_path_dict:
			if key > max_frame:
				max_frame = key

		
		for frame in range(min_frame, max_frame + 1):

			print("\tframe %d" % frame)
			#frame = 259 #225
			frame_path = frame_path_dict[frame]
			image_np = load_image_np(frame_path)
			image_info = image_info_bundel[frame_path]

			try:

				person_box = get_high_score_box(image_info, 'person')
				person_mark = get_person_mark(person_box)

				ball_box = get_high_score_box(image_info, 'basketball')
				ball_mark = get_ball_mark(ball_box)
				ball_radius = get_ball_radius(ball_box)

				#get person and box histories
				person_boxes.append(person_box)
				ball_boxes.append(ball_box)

				#add to history - todo: get all from person and box history
				person_marks.append(person_mark)
				ball_marks.append(ball_mark)
				ball_radii.append(ball_radius)

				outside_mark = get_ball_outside_mark(person_box, ball_box)

				#test draw boxes
				#image_np = draw_all_boxes_image_np(image_np, image_info)
				#test draw basketball outline
				image_np = draw_circle(image_np, ball_mark, radius=ball_radius, color=(0,255,0), thickness=2)

				
				#draw history with decreasing brighness
				#for l in range(len(ball_marks)):

				#	for j in range(l-1):
				#		k = j+1
				#		draw_person_ball_connector(image_np, ball_marks[j], ball_marks[k])

				#	image_np = draw_circle(image_np, ball_marks[l], color=(200,200,255))
				#	#image_np = draw_circle(image_np, person_marks[l], color=(200,200,255))

				#	#image_np = draw_circle(image_np, ball_marks[l], radius=ball_radii[l], color=(0,100,0), thickness=2)

				#test draw person mark
				image_np = draw_circle(image_np, person_mark)

				#test draw ball mark
				image_np = draw_circle(image_np, ball_mark)

				#test draw person ball connector
				#image_np = draw_person_ball_connector(image_np, person_mark, ball_mark)
				image_np = draw_person_ball_connector(image_np, person_mark, outside_mark)

				write_frame_for_accuracy_test(output_image_directory, frame, image_np)
			except:
				pass
		
		"""



		"""
		person_areas = [box_area(box) for box in person_boxes]
		basketball_areas = [box_area(box) for box in ball_boxes]
		print("person boxes: " + str(person_areas))
		print("basketball boxes: " + str(basketball_areas))

		person_shs = [height_squared(box) for box in person_boxes]
		print(person_shs)
		"""

		
		"""
		# write video
		output_frame_paths_dict = get_frame_path_dict(output_image_directory)
		ordered_frame_paths = []
		input_frame_paths_dict = frame_path_dict	#for errors
		for frame in range(min_frame, max_frame + 1):
			try:
				ordered_frame_paths.append(output_frame_paths_dict[frame])
			except: 
				ordered_frame_paths.append(input_frame_paths_dict[frame])

		output_video_filepath = 'output_video/shot_%d_tracking_3.mp4' % i
		write_mp4_video(ordered_frame_paths, 'JPEG', output_video_filepath)
		"""


		#
		#	find distance between highest score player and highest score ball
		#

		#size of basketball : number of pixels
		#NBA is 29.5 inches in circumference
		#diameter = 29.5/(3.14) = 9.4 in

		"""
		# FLAGS to detect state

		PERSON_WITH_BALL = bool


		idea:
			anaylize scene purley in terms of basketball box and person box

			scale and center around person? how to scale and center


		"""


		"""
		#tmp edit first saved image info bundel paths add "frames_shot_1"
		new_image_info_bundel = {}
		for path, image_info in image_info_bundel.items():
			#add_dir = "/frames_shot_1"
			#prefix, filename = os.path.split(path)
			#new_path = os.path.join(prefix + add_dir, filename)
			new_image_info = image_info.copy()
			new_image_info['image_folder'] = "frames_shot_1"
			new_image_info_bundel[path] = new_image_info

		print(new_image_info_bundel)

		with open(image_info_bundel_filepath, 'w') as file:
			file.write(json.dumps(new_image_info_bundel))
		"""

	


