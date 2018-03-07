#!/usr/bin/env python

"""

docstring format:

\"\"\"
This is a reST style.

:param param1: this is a first param
:param param2: this is a second param
:returns: this is a description of what is returned
:raises keyError: raises an exception
\"\"\"

[TODO]:
	- detect if video is stable
	- display velocity
"""

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
from mpl_toolkits import mplot3d                            #   3d plotting
from scipy import stats                                     #   error rvalue/linear regression
from piecewise.regressor import piecewise                   #   piecewise regression
from piecewise.plotter import plot_data_with_regression

# my lib
#from image_evaluator.src import image_evaluator

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#   note: #(left, right, top, bottom) = box

def pixel_movement_between_frames(frame_image1, frame_image2):
	"""
	:param frame_image1: first numpy array image to compare
	:param frame_image2: second nnumpy array image to compare
	:return: returns boolean True if there is pixel movement between frames. Note: ignores object motion between frames
	"""
	pass

def is_video_stable(input_frames_directory):
	"""
	:param input_frames_directory: String of path to directory containing video frame images
	:return: boolean True if there is no pixel movement.
	"""
	pass

#
# Test accuracy by writing new images
#

#ext = extension
#source: http://tsaith.github.io/combine-images-into-a-video-with-python-3-and-opencv-3.html
def write_mp4_video(ordered_image_paths, ext, output_mp4_filepath):
	"""
	:param ordered_image_paths: array of image path strings to combine into mp4 video file
	:param ext: NOT USED
	:param output_mp4_filepath: output file name without extension
	"""

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
	"""
	:param output_directory_path: String of path to directory to write numpy array image to with filename "frame_%s.JPEG" where %s is frame passed as parameter. Path string can be relative or absolute.
	:param frame: int, frame number
	:param image_np: numpy array image to write to file
	"""
	# if image output directory does not exist, create it
	if not os.path.exists(output_directory_path): os.makedirs(output_directory_path)

	image_file_name = "frame_%d.JPEG" % frame 
	output_file = os.path.join(output_directory_path, image_file_name)
	
	cv2.imwrite(output_file, image_np)  #BGR color

	"""
	#fix color
	image = Image.fromarray(image_np, 'RGB')
	image.save(output_file)
	logger.info("wrote %s to \n\t%s" % (image_file_name, output_file))
	"""

#list of 4 coordanates for box
def draw_box_image_np(image_np, box, color=(0,255,0), width=3):
	"""
	:param image_np: numpy array image
	:param box: tuple (x1,x2,y1,y2)
	:param color: tuple (R,G,B) color to draw, defualt is (0,255,0)
	:param width: int, width of rectangle line to draw, default is 3
	:return: returns numpy array of image with rectangle drawn
	"""
	(left, right, top, bottom) = box
	cv2.rectangle(image_np,(left,top),(right,bottom),color,3)
	return image_np

def draw_all_boxes_image_np(image_np, image_info):
	"""
	:param image_np: numpy array image
	:param image_info: image_info object with objects containing "box" tuples to draw to numpy array image
	:return: returns numpy array image with all recatangles drawn
	"""
	for item in image_info['image_items_list']:
		draw_box_image_np(image_np, item['box'])
	return image_np

def get_category_box_score_tuple_list(image_info, category):
	"""
	:param image_info: image/frame info object
	:param category: category/class string
	:returns: returns list of tupples containing objects "box" and "score" of all objects with matching "class"
	"""
	score_list = []
	box_list = []
	for item in image_info['image_items_list']:
			if item['class'] == category:
					box_list.append(item['box'])
					score_list.append(item['score'])
	return list(zip(score_list, box_list))

def get_high_score_box(image_info, category, must_detect=True):
	"""
	:param image_info: image/frame info object
	:param category: category/class string
	:param must_detect: must_detect boolean. Will throw error if value is True and no matching class is found. Will return None otherwise. default is True
	:returns: "box" of object with highest "score" of selected "class"
	"""
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
	"""
	:param person_box: tupple (x1,x2,y1,y2)
	:returns: Point (x,y) where x is half the person box's width, and y is 3/4 the person box's height
	"""
	# 3/4 height, 1/2 width
	(left, right, top, bottom) = person_box
	width = int((right - left)/2)
	x = left + width
	height = int((bottom - top)*float(1.0/4.0))
	y = top + height
	return (x,y)

def get_ball_mark(ball_box):
	"""
	:param ball_box: tupple (x1,x2,y1,y2)
	:returns: Point (x,y) where x and y are half the ball box's width and height
	"""
	# 1/2 height, 1/2 width
	(left, right, top, bottom) = ball_box
	width = int((right - left)/2)
	x = left + width
	height = int((bottom - top)/2)
	y = top + height
	return (x,y)

def get_angle_between_points(mark1, mark2):
	"""
	:param mark1: Point/(x,y)
	:param mark2: Point?(x,y)
	:returns: Angle between points
	"""
	x1, y1 = mark1
	x2, y2 = mark2
	radians = math.atan2(y1-y2,x1-x2)
	return radians

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

def get_ball_outside_mark(person_box, ball_box):
	"""
	:param person_box: tupple (x1,x2,y1,y2)
	:param ball_box: tupple (x1,x2,y1,y2)
	:returns: Return Point located on balls radial surface closest to the person box passed as paramenter.
	"""
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
	"""
	:param box: tuple (x1,x2,y1,y2)
	:returns: area of box
	"""
	(left, right, top, bottom) = box
	return (right-left) * (bottom-top)

def height_squared(box):
	"""
	:param box: tuple (x1,x2,y1,y2)
	:returns: height of box squared
	"""
	(left, right, top, bottom) = box
	return (bottom-top)**2

#center (x,y), color (r,g,b)
def draw_circle(image_np, center, radius=2, color=(0,0,255), thickness=10, lineType=8, shift=0):
	"""
	:param image_np: numpy array image
	:param center: tuple (x,y) center of circle to draw
	:param radius: int, radius of circle to draw
	:param color: tuple (R,G,B) color to draw, defualt is (0,0,255)
	:param thickness: int, thickness of line to draw, default is 10
	:param lineType: opencv parameter
	:param shift: opencv parameter
	:return: returns numpy array of image with circle drawn
	"""
	cv2.circle(image_np, center, radius, color, thickness=thickness, lineType=lineType, shift=shift)
	return image_np

def draw_person_ball_connector(image_np, person_mark, ball_mark, color=(255,0,0)):
	"""
	:param image_np: image to draw line onto
	:param person_mark: tuple (x,y) of one line endpoint
	:param ball_mark: tuple (x,y) of one line endpoint
	:param color: tuple (R,G,B) color of line, default is (255,0,0)
	:return: numpy array image with line drawn
	"""
	lineThickness = 7
	cv2.line(image_np, person_mark, ball_mark, color, lineThickness)
	return image_np

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


def load_image_np(image_path):
	"""
	:param image_path: string of path to image, relative or non relative
	:returns: returns numpy array of image
	"""
	#non relitive path
	script_dir = os.path.dirname(os.path.abspath(__file__))
	image = Image.open(os.path.join(script_dir, image_path))
	(im_width, im_height) = image.size
	image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

	return image_np

def filter_minimum_score_threshold(image_info_bundel, min_score_thresh):
	"""
	:param image_info_bundel: image_info_bundel object
	:param min_score_thresh: Minimum score threshold of objects not to filter out of returned image_info_bundel
	:returns: return filtered image_info_bundel object
	"""
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
	"""
	:param image_info_bundel: image_info_bundel object
	:param selected_categories_list: list of strings of categories/classes to keep in returned image_info_bundel object
	:returns: return filtered image_info_bundel object
	"""
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
	"""
	:param image_directory_dirpath: String of directory path to images to be evaluated using image_evaluator
	:param image_boolean_bundel_filepath: String of file path to output image_boolean_bundel file, creates if does not exist
	:param image_info_bundel_filepath: String of file path to output image_info_bundel file, creates if does not exist
	:param model_list: list of models to use in evaluation format: {'name' : 'model name', 'use_display_name' : Boolean, 'paths' : {'frozen graph': "path/to/frozen/interfernce/graph", 'labels' : "path/to/labels/file"}}
	:param bool_rule: boolean statement string for evaluations using class names, normal boolean logic python syntax and any() and num() methods
	"""

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
	"""
	:param image_info_bundel_filepath: String to image_info_bundel_file json file
	:returns: image_info_bundel python dictornary
	"""
	with open(image_info_bundel_filepath) as json_data:
		d = json.load(json_data)
	return d

def get_frame_path_dict(video_frames_dirpath):
	"""
	:param video_frames_dirpath: String path to directory full of frame images with name format "frame_i.JPEG"
	:return: python dictonary of frame_path_dict format {frame_number:"path/to/image, ...}
	"""

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
	"""
	:param frame_path_dict: Python dictonary format {frame_number:"path/to/image, ...}
	:return: minimum frame, maximum frame, and a boolean of whether or not a continous range exists
	"""
	frames, paths = zip(*frame_path_dict.items())

	min_frame, max_frame = min(frames), max(frames)
	continuous = set(range(min_frame,max_frame+1)) == set(frames)

	return min_frame, max_frame, continuous 

def frame_directory_to_video(input_frames_directory, output_video_file):
	"""
	:param input_frames_directory: String of path to directory of images 'frame_i.JPEG" format currently supported. Frame range must be continous. Path can be absolute or relative. 
	:param output_video_file: String of path of output directory
	"""
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
			#draw_box_image_np(rgb_blank_image, person_box)
			#draw_circle(rgb_blank_image, (px, py))
			#draw_box_image_np(rgb_blank_image, ball_box) #ball box
			#draw_circle(rgb_blank_image, (bx, by)) #ball circle
			#draw_person_ball_connector(rgb_blank_image, (px,py), (bx,by)) #draw connectors


			#iou overlap
			if iou(person_box, ball_box) > 0:

				#
				#old coordinate drawings
				#

				#ball
				draw_circle(rgb_blank_image, (bx, by), color=(0,0,255)) #mark
				draw_circle(rgb_blank_image, (bx, by), radius=ball_radius, color=(0,0,255), thickness=5) #draw ball
				draw_person_ball_connector(rgb_blank_image, (px, py), (bx, by), color=(0,0,255)) # connector

				#person
				draw_circle(rgb_blank_image, (px, py), color=(0,0,255))
				#draw_box_image_np(rgb_blank_image, person_box, color=(0,0,255))

				#
				#new coordinate drawings
				#

				#ball
				#draw_circle(rgb_blank_image, new_ball_mark, color=(0,255,0))   #mark
				#draw_circle(rgb_blank_image, new_ball_mark, radius=ball_radius, color=(0,255,0), thickness=5) #draw ball
				#draw_person_ball_connector(rgb_blank_image, center, new_ball_mark, color=(0,255,0)) # connector

				#person
				#draw_circle(rgb_blank_image, center, color=(0,255,0))
				#draw_box_image_np(rgb_blank_image, new_person_box, color=(0,255,0))

			else:

				#
				#old coordinate drawings
				#

				#ball
				draw_circle(rgb_blank_image, (bx, by), color=(0,0,255)) #mark
				draw_circle(rgb_blank_image, (bx, by), radius=ball_radius, color=(0,0,255), thickness=5) #draw ball
				#draw_person_ball_connector(rgb_blank_image, (px, py), (bx, by), color=(0,255,0)) # connector

				#person
				#draw_circle(rgb_blank_image, (px, py), color=(0,255,0))
				#draw_box_image_np(rgb_blank_image, person_box, color=(0,255,0))

				#
				#new coordinate drawings
				#

				#ball
				#draw_circle(rgb_blank_image, new_ball_mark, color=(0,0,255))   #mark
				#draw_circle(rgb_blank_image, new_ball_mark, radius=ball_radius, color=(0,0,255)) #ball
				#draw_person_ball_connector(rgb_blank_image, center, new_ball_mark, color=(0,0,255)) #connector

				#person
				#draw_circle(rgb_blank_image, center, color=(0,0,255))
				#draw_box_image_np(rgb_blank_image, new_person_box, color=(0,0,255))

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

			frame_image = cv2.imread(frame_path)    #read image
			image_np = cycle_function(frame_image, image_info)

			if apply_history:

				# TODO: fix weights
				for i in range(frame, min_frame, -1):
					alpha = 0.1
					beta = 0.1
					gamma = 0.5
					i_frame_path = frame_path_dict[i]
					i_image_info = image_info_bundel[i_frame_path]
					i_frame_image = cv2.imread(i_frame_path)    #read image
					next_image_np = cycle_function(i_frame_image, i_image_info)
					image_np = cv2.addWeighted(image_np,alpha,next_image_np,beta,gamma)

			# write images
			#write_frame_for_accuracy_test(output_frames_directory, frame, image_np)

		# write video
		frame_directory_to_video(output_frames_directory, output_video_file)
	else:
		logger.error("not continuous")

# source: https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy
def group_consecutives(vals, step=1):
	"""
	:param vals: list of integers
	:param step: step size, default 1
	:return: list of consecutive lists of numbers from vals (number list).
	"""
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

# frame_info_bundel to frame_path_dict
def frame_info_bundel_to_frame_path_dict(input_frame_info_bundel):
	"""
	:param input_frame_info_bundel: frame_info_bundel object. Must be only one directory containing all frames
	:return: frame_path_dict python dictonary format {frame_number : "path/to/frame", ...}
	"""
	#find unique input_frame_directories, assert there is only 1
	unique_input_frame_directories = list(set([os.path.split(frame_path)[0] for frame_path, frame_info in input_frame_info_bundel.items()]))
	assert len(unique_input_frame_directories) == 1

	# return frame_path_dict
	return get_frame_path_dict(unique_input_frame_directories[0])



#
#                           ball collected data points matrix (ball_cdpm)
#
# format 2:
# columns: frame, x1, x2, y1, y2, ball state / iou bool
#
# enum keys: 
#           'frame column', 'x1 column', 'x2' column, 'y1 column', 'y2 column', 'ball state column'     #columns 
#           'no data', 'free ball', 'held ball'                                                         #ball states
#
# "essentially an alternative representation of image_info_bundel"
# to access: frames, xs, ys, state = ball_cdpm.T

def create_ball_cdpm( ball_cdpm_enum, frame_info_bundel):

	# get frame path dict
	frame_path_dict = frame_info_bundel_to_frame_path_dict(frame_info_bundel)

	# get minimum and maximum frame indexes
	min_frame, max_frame, continuous = min_max_frames(input_frame_path_dict)

	#   Matrix - fill with no data
	num_rows = (max_frame + 1) - min_frame
	num_cols = 6                                    # frame, x1, x2, y1, y2, state (iou bool)
	ball_cdpm = np.full((num_rows, num_cols), ball_cdpm_enum['ball_states']['no_data'])

	# iou boolean lambda function for 'ball mark x column'
	get_ball_state = lambda person_box, ball_box : ball_cdpm_enum['ball_states']['held_ball'] if (iou(person_box, ball_box) > 0) else ball_cdpm_enum['ball_states']['free_ball']

	#                   Fill ball collected data points matrix (ball_cdpm)
	#
	#                   'frame', 'x1', 'x2', 'y1', 'y2', 'state'

	index = 0
	for frame in range(min_frame, max_frame + 1):
		frame_path = frame_path_dict[frame]
		frame_info = frame_info_bundel[frame_path]

		#get frame ball box and frame person box
		frame_ball_box = get_high_score_box(frame_info, 'basketball', must_detect=False)
		frame_person_box = get_high_score_box(frame_info, 'person', must_detect=False)

		# frame number column ['column']['frame']
		ball_cdpm[index,ball_cdpm_enum['cdpm_columns']['frame']] = frame

		#ball box coulms ['column'][i] for i in 'x1', 'x2', 'y1', 'y2'
		if (frame_ball_box is not None):
			ball_cdpm[index,ball_cdpm_enum['cdpm_columns']['x1']] = frame_ball_box[0]
			ball_cdpm[index,ball_cdpm_enum['cdpm_columns']['x2']] = frame_ball_box[1]
			ball_cdpm[index,ball_cdpm_enum['cdpm_columns']['y1']] = frame_ball_box[2]
			ball_cdpm[index,ball_cdpm_enum['cdpm_columns']['y2']] = frame_ball_box[3]

		#ball state/iou bool column ['column']['state']
		if (frame_ball_box is not None) and (frame_person_box is not None):
			ball_cdpm[index,ball_cdpm_enum['cdpm_columns']['state']] = get_ball_state(frame_person_box, frame_ball_box)

		index += 1

	# return matrix
	return ball_cdpm

def ball_cdpm_boxes_to_marks(ball_cdpm_enum_old, ball_cdpm_enum_new, ball_cdpm):
	"""
	:param ball_cdpm_enum_old: python dictory specifying the coulmn indices of ['cdpm_columns']: 'frame', 'x1', 'x2', 'y1', 'y2', 'state', and the number map for each state ['ball_states']: 'no_data', 'held_ball', 'free_ball'
	:param ball_cdpm_enum_new: python dictory specifying the coulmn indices of ['cdpm_columns']: 'frame', 'x', 'y', 'state', and the number map for each state ['ball_states']: 'no_data', 'held_ball', 'free_ball'
	:param ball_cdpm: ball_cdpm with columns:  'frame', 'x1', 'x2', 'y1', 'y2', 'state'
	:return: ball_cdpm with columns:  frame, x, y, state - where (x,y) is center point of basketball box or "ball_mark"
	"""

	# old cdpm
	frames = ball_cdpm[:,ball_cdpm_enum_old['cdpm_columns']['frame']]
	x1s = ball_cdpm[:,ball_cdpm_enum_old['cdpm_columns']['x1']]
	x2s = ball_cdpm[:,ball_cdpm_enum_old['cdpm_columns']['x2']]
	y1s = ball_cdpm[:,ball_cdpm_enum_old['cdpm_columns']['y1']]
	y2s = ball_cdpm[:,ball_cdpm_enum_old['cdpm_columns']['y2']]
	states = ball_cdpm[:,ball_cdpm_enum_old['cdpm_columns']['state']]

	# new cdpm additions
	num_rows = ball_cdpm.shape[0]
	xs = np.full(num_rows, ball_cdpm_enum_old['ball_states']['no_data'])
	ys = xs.copy()
	
	# find ball marks and polulate new column arrays
	indxs = range(0,num_rows)
	for i,x1,x2,y1,y2 in zip(indxs,x1s,x2s,y1s,y2s):
		ball_box = (x1,x2,y1,y2)
		x, y = get_ball_mark(ball_box)
		xs[i] = x
		ys[i] = y

	# create new cdpm with marks instead of boxes
	new_ball_cdpm = np.array([frames, xs, ys, states]).T
	
	# arrage columns according to ball_cdpm_enum_new parameter
	new_frames_index = ball_cdpm_enum_new['cdpm_columns']['frame']
	new_xs_index = ball_cdpm_enum_new['cdpm_columns']['x']
	new_ys_index = ball_cdpm_enum_new['cdpm_columns']['y']
	new_states_index = ball_cdpm_enum_new['cdpm_columns']['state']
	column_indices = np.array([new_frames_index, new_xs_index, new_ys_index, new_states_index])
	
	#rearange
	new_ball_cdpm = new_ball_cdpm.T
	new_ball_cdpm = new_ball_cdpm[column_indices].T

	# return
	return new_ball_cdpm



	

#
#                           ball collected data points matrix (ball_cdpm)
#
# format 1:
# columns: frame, x, y, ball state / iou bool
#
# enum keys: 
#           'frame column', 'ball mark x column', 'ball mark y column', 'ball state column'     #columns 
#           'no data', 'free ball', 'held ball'                                                 #ball states
#
# "essentially an alternative representation of image_info_bundel"
# to access: frames, xs, ys, state = ball_cdpm.T
#

def get_ball_cdpm( ball_cdpm_enum, input_frame_path_dict, image_info_bundel):

		# get minimum and maximum frame indexes
		min_frame, max_frame, continuous = min_max_frames(input_frame_path_dict)

		#   Matrix - fill with no data

		num_rows = (max_frame + 1) - min_frame
		num_cols = 4                        # frame, ballmark x, ballmark y, ball state (iou bool)
		ball_cdpm = np.full((num_rows, num_cols), ball_cdpm_enum['no data'])

		# iou boolean lambda function for 'ball mark x column'
		get_ball_state = lambda person_box, ball_box : ball_cdpm_enum['held ball'] if (iou(person_box, ball_box) > 0) else ball_cdpm_enum['free ball']

		#                   Fill ball collected data points matrix (ball_cdpm)
		#
		#                   'frame', 'ballmark x', 'ballmark y', 'ball state'

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


# get list of shot frame ranges
def find_shot_frame_ranges(frames_info_bundel, std_error_threshold=0.9, single_data_point_shots=False):
	"""
	:param frames_info_bundel: frames_info_bundel object to extract shot ranges from
	:param std_error_threshold: Standard error threshold for x axis regression line before performing peicewise regression and returning first line segment as range
	:param single_data_point_shots: Return single data point shot frame ranges boolean, default is set to False
	:return: list of lists [[shot_frame_range_i_start, shot_frame_range_i_stop], ...]
	"""
	#
	#   Find shot frame ranges
	#   Mock 2: Assertions: Stable video, 1 person, 1 ball

	#
	#   Build  Format 2 ball data points matrix  (ball_cdpm)
	#                                                   columns: frame, ball box x1, ball box x2, ball box y1, ball box y2, ball state
	#

	ball_cdpm_enum = {
		'ball_states' : {
			'no_data' : -1,
			'free_ball' : 1,
			'held_ball' : 0
		},
		'cdpm_columns' : {
			'frame' : 0,
			'x1' : 1,
			'x2' : 2,
			'y1' : 3,
			'y2' : 4,
			'state' : 5,
		}
	}

	ball_cdpm = create_ball_cdpm(ball_cdpm_enum, frames_info_bundel)

	#
	# Break ball data points matrix into multiple submatrices (free_ball_cbdm's) where ball is free
	#               (break around held ball datapoints)
	#

	# cut ball_cdpm at frames with ball state column value == 'held ball', leaving only free ball datapoints in an array of matrices
	free_ball_cbdm_array = ball_cdpm[ball_cdpm[:, ball_cdpm_enum['cdpm_columns']['state']] != ball_cdpm_enum['ball_states']['held_ball'], :]    # extract all rows with the ball state column does not equal held ball
	free_ball_cbdm_array = group_consecutives_by_column(free_ball_cbdm_array, ball_cdpm_enum['cdpm_columns']['frame'])                      	# split into seperate matrices for ranges


	#
	# Find shot frame ranges
	#
	shot_frame_ranges = []

	#   for each free_ball_cbdm (sub matrix)
	for i in range(len(free_ball_cbdm_array)):
		possible_trajectory_coordinates = free_ball_cbdm_array[i]       # 'possible' ball trajectory coordinates, ranges without held ball states tagged by model

		# extract 'known' ball trajectory coordinates, tagged by model
		#remove missing datapoints 
		known_trajectory_points = possible_trajectory_coordinates[possible_trajectory_coordinates[:, ball_cdpm_enum['cdpm_columns']['state']] != ball_cdpm_enum['ball_states']['no_data'], :]   # extract all rows where there is data 
		kframes, kx1s, kx2s, ky1s, ky2s,  kstate = known_trajectory_points.T

		kball_boxes = list(zip(kx1s, kx2s, ky1s, ky2s))
		kball_marks = [get_ball_mark(bb) for bb in kball_boxes]

		#enure known ball trajectory has more than 1 data point
		if len(kframes) > 1:

			#find average x and y values
			kxs, kys = zip(*kball_marks)

			#
			#                       peicewise linear regression
			#
			#   Apply if x std error is abole threshold
			#   Apply peicewise linear regression to x values.
			#   x values should change linearly if ball is in free flight (ignoring air resistance).
			#   Use peicewise linear regression to find the point at which the free flying ball hits another object/changes its path
			#   Find point of intersection for seperate regression lines to find final frame of shot ball trajectory
			#

			# test linear regression std error for x values
			slope, intercept, r_value, p_value, std_err = stats.linregress(kframes, kxs)

			logger.info("\n\nSTD Error for Regression: %s" % std_err)

			start_frame, stop_frame = kframes[0], kframes[-1]
			shot_frame_range = [start_frame, stop_frame]

			# apply peicewise linear regression only if x std error is above threshold 
			# take first line segment start and stop points as shot_frame_range
			if std_err >= std_error_threshold:

				logger.info("Applying peicewise regression\n")

				#peicewise model
				model = piecewise(kframes, kxs)
				start_frame, stop_frame, coeffs = model.segments[0]     # find start and stopping frame (start_frame, stop_frame) from first line segment, and line formula
				shot_frame_range = [start_frame, stop_frame]

			shot_frame_ranges.append(shot_frame_range)

		# single datapoint shot frame range - [start/stop frame, start/stop frame]
		if (len(kframes) == 1) and single_data_point_shots:
			shot_frame_range = [kframes[0], kframes[0]]
			shot_frame_ranges.append(shot_frame_range)

	return shot_frame_ranges

def known_boxes_in_frame_range(frame_info_bundel, shot_frame_range, category):
	"""
	:param frame_info_bundel: frames_info_bundel object to extract boxes from
	:param shot_frame_range: frame range to extract to extract boxes from
	:param category: String category/class to extract high score boxes from 
	:return: [box, frame] where box is a tuple (x1, x2, y1, y2)
	"""
	# get frame path dict
	frame_path_dict = frame_info_bundel_to_frame_path_dict(frame_info_bundel)

	# minimum and maximum frames
	min_frame, max_frame = shot_frame_range[0], shot_frame_range[1]

	# Find ball collected boxes
	known_boxes_and_frames = []
	for frame in range(min_frame, max_frame + 1):
		frame_path = frame_path_dict[frame]
		frame_info = frame_info_bundel[frame_path]

		#get frame high scroe ball box
		frame_ball_box = get_high_score_box(frame_info, 'basketball', must_detect=False)

		if (frame_ball_box is not None):
			known_boxes_and_frames.append([frame_ball_box, frame])

	return known_boxes_and_frames

def find_ball_regression_formulas(frame_info_bundel, shot_frame_range, adjust_yvalues=True):
	"""
	:param frames_info_bundel: frames_info_bundel object to extract regression formulas from
	:param shot_frame_range: frame range to extract regression formulas from
	:return: regression polynomial coeffiecnts list [pxs,pys]. format pis: [(coeff 0)frame^0, (coeff 1)frame^1, ...] -- (np.polyfit)
	"""
	# [TODO] Clean this shit
	# note cannot handle frame ranges of single value

	# get xs, ys, radiis known datapoints in frame range

	# get known boxes in frame range
	ball_boxes, frames = zip(*known_boxes_in_frame_range(frame_info_bundel, shot_frame_range, 'basketball'))	#tuples
	
	# get known ball marks
	ball_marks = [get_ball_mark(bb) for bb in ball_boxes]

	#find average x and y values
	xs, ys = zip(*ball_marks) 	#tuples

	# find ball radii
	ball_radii = [get_ball_radius(bb, integer=False) for bb in ball_boxes]

	# normalize radii for change only
	# normalize to first radii
	normalized_ball_radii = [r/ball_radii[0] for r in ball_radii]

	# zs_distance_change_coeff - 1/normalized ball radii. this represents the balls distance change from its startposition at the origin
	# greater than 1 is farther away, 2 is twice as far away
	zs_distance_change_coeff = [1/r for r in normalized_ball_radii]

	# find regression formula then scale to balls distance away
	pzs_change_coeff = np.polyfit(frames, zs_distance_change_coeff, 1)
	total_shot_frames = np.linspace(frames[0], frames[-1], frames[-1]-frames[0])
	#zs_change_coeffs = np.polyval(pzs_change_coeff, total_shot_frames)
	zs_change_coeffs = np.polyval(pzs_change_coeff, frames)

	# ys adjuseted
	# y_adjusted = y/z_change_coeff_matched_range
	# 
	ys_adjusted = [y/zcc for y,zcc in zip(ys, zs_change_coeffs)]

	# find x regression polynomial coeffiecnts
	#xs - degreen 1 regression fit 
	pxs = np.polyfit(frames, xs, 1)

	# find y regreesion polynomial coeffiecents - currently do not take into account z corrections
	#ys - degreen 2 regression fit 
	pys = np.polyfit(frames, ys, 2)

	if adjust_yvalues:
		# find pys with z correction
		pys = np.polyfit(frames, ys_adjusted, 2)

	# return polynomial coefficents
	return [pxs, pys]

def find_normalized_ball_regression_formulas(frame_info_bundel, shot_frame_range, adjust_yvalues=True, amplify_zslope=True, return_radii=False):
	"""
	:param frames_info_bundel: frames_info_bundel object to extract regression formulas from
	:param shot_frame_range: frame range to extract regression formulas from
	:param adjust_yvalues:
	:param amplify_zslope:
	:param return_radii:
	:return: normalized regression polynomial coeffiecnts to balls radius in pixels list [pxs,pys]. format pis: [(coeff 0)frame^0, (coeff 1)frame^1, ...] -- (np.polyfit)
	"""

	# NOTE: cannot handle frame ranges of single value

	#
	# 	Find normalized polynomial coefficients describing shot trajectories for x,y and z axes within given frame range
	#	
	#		* use trends in basketball radii to determine z axes function
	#		* adjust y function using z function
	#		* normalize functions to the radius of the basketball in pixels


	#
	#	Retrieve x, y, and radius data points identified by model within the given frame range
	#
	#		* xs, ys, radii, frames, start_frame, stop_frame

	# get boxes, frames, start_frame, and stop_frame in given frame range
	ball_boxes, frames = zip(*known_boxes_in_frame_range(frame_info_bundel, shot_frame_range, 'basketball'))
	start_frame, stop_frame = frames[0], frames[-1]

	# get average x and y values within given frame range (ball_marks)
	ball_marks = [get_ball_mark(bb) for bb in ball_boxes]
	xs, ys = zip(*ball_marks)

	# get ball radii within frame range
	radii = [get_ball_radius(bb, integer=False) for bb in ball_boxes]

	#
	#	Find z axis normalized polynomial coefficients
	#
	#		* normalize basketball radii to start_frames radius
	#		* find regression function describing trends in the changing normalized basketball radius sizes
	#		# amplify normalized radii size regression function to find normalized z function
	#
	#		Justification : If the balls radius is twice as far away it will be half the size


	# normalize basketball radii to start_frames radius
	norm_radii = [r/radii[0] for r in radii]

	# find regression function describing trends in the changing normalized basketball radius sizes
	slope, intercept, r_value, p_value, std_err = stats.linregress(frames, norm_radii)

	# amplify normalized radii size regression function to find normalized z function
	#	* amplify slope of normalized radii size regression function 
	#	* damp slope of normalized radii size regression function with r_value
	amplified_slope = (slope)*((abs(slope) + math.pi)**3)*(r_value**3)

	# normalized polynomial coefficients for z axis function
	pzs_norm = [slope, intercept] 										# final
	pzs_norm_amplified = [amplified_slope, intercept] 					# final (if amplify_zslope)

	#
	#	Find y axis normalized polynomial coefficients
	#
	#		* adjust y data points using z function:
	#													formula: yi_adjusted = yi * zi_norm
	#		* normalize adjusted y datapoints to basketball radius in pixels
	#		* find polynomial coefficients for second degree regression fit of normalized and adjusted y values

	# adjust y data points using z function
	zs_norm = np.polyval(pzs_norm, frames)

	if adjust_yvalues:
		ys = [y*z for y,z in zip(ys, zs_norm)]

	# normalize adjusted y datapoints to basketball radius in pixels
	ys_norm = [y/r for y,r in zip(ys, radii)]

	# find polynomial coefficients for second degree regression fit of normalized and adjusted y values
	pys_norm = np.polyfit(frames, ys_norm, 2)					# final

	#
	#	Find x axis normalized polynomial coefficients
	#
	#		* normalize adjusted x datapoints to basketball radius in pixels
	#		* find polynomial coefficients for first degree regression fit of normalized and adjusted x values

	# normalize adjusted x datapoints to basketball radius in pixels
	xs_norm = [x/r for x,r in zip(xs, radii)] 

	# find polynomial coefficients for first degree regression fit of normalized and adjusted x values
	pxs_norm = np.polyfit(frames, xs_norm, 1)				# final

	#
	# return normalized polynomial coefficents
	#

	if amplify_zslope:
		if not return_radii:
			return [pxs_norm, pys_norm, pzs_norm_amplified]
		return [pxs_norm, pys_norm, pzs_norm_amplified, radii]

	if not return_radii:
		return [pxs_norm, pys_norm, pzs_norm]
	return [pxs_norm, pys_norm, pzs_norm, radii]

def pixel_shot_position_vectors(frame_info_bundel, shot_frame_range, extrapolate=False):
	"""
	:param frame_info_bundel: frames_info_bundel object to extract pixel position vectors from
	:param shot_frame_range: frame range to extract pixel position vectors from
	:param extrapolate: boolean default False, extrapolate trajectories from known datapoints filling in unkown frames
	:return: return matrix of pixel position vectors with frame, x,y as columns. 
	"""
	start_frame, stop_frame = shot_frame_range[0], shot_frame_range[1]
	shot_frames = np.linspace(start_frame, stop_frame, stop_frame-start_frame+1)

	if not extrapolate:
		#
		#	Return raw data identified by model in pixel matrix, mask frames with no data points in given frame range
		#

		ball_cdpm_enum_old = {
			'ball_states' : {
				'no_data' : -1,
				'free_ball' : 1,
				'held_ball' : 0
			},
			'cdpm_columns' : {
				'frame' : 0,
				'x1' : 1,
				'x2' : 2,
				'y1' : 3,
				'y2' : 4,
				'state' : 5,
			}
		}

		ball_cdpm_enum = {
			'ball_states' : {
				'no_data' : -1,
				'free_ball' : 1,
				'held_ball' : 0
			},
			'cdpm_columns' : {
				'frame' : 0,
				'x' : 1,
				'y' : 2,
				'state' : 3,
			}
		}

		# columns: frame, x1, x2, y1, y2, state 
		ball_cdpm = create_ball_cdpm(ball_cdpm_enum_old, frame_info_bundel)

		# convert to : columns: frame, x, y, state 
		ball_cdpm = ball_cdpm_boxes_to_marks(ball_cdpm_enum_old, ball_cdpm_enum, ball_cdpm)

		# trim rows to frame range
		frame_column_index = ball_cdpm_enum['cdpm_columns']['frame']
		after_indices = np.where(ball_cdpm[:,frame_column_index] >= start_frame)[0]
		before_indices = np.where(ball_cdpm[:,frame_column_index] <= stop_frame)[0]
		start_index = after_indices[0]
		stop_index = before_indices[-1]
		ball_cdpm = ball_cdpm[start_index:stop_index,:]

		# trim ball state column
		ball_cdpm = np.delete(ball_cdpm, ball_cdpm_enum['cdpm_columns']['state'], 1)

		# trim frame column
		ball_cdpm = np.delete(ball_cdpm, ball_cdpm_enum['cdpm_columns']['frame'], 1)

		# mask rows with no datapoint
		x_column_index = ball_cdpm_enum['cdpm_columns']['x']
		row_mask_indices = np.where(ball_cdpm[:,x_column_index] == ball_cdpm_enum['ball_states']['no_data'])[0]

		h,w = ball_cdpm.shape
		mask_np = np.zeros((h, w), dtype=int)

		for r in range(h+1):
			if r in row_mask_indices:
				mask_np[r] = 1
		
		# create masked pixel matrix
		masked_pixel_matrix = np.ma.array(ball_cdpm, mask=mask_np)

		# return
		return masked_pixel_matrix

	else:
		#
		# return extrapolated pixel matrix
		#		* y values not adjusted
		#		* no mask/empty datapoints in frame range
		
		pxs, pys = find_ball_regression_formulas(frame_info_bundel, shot_frame_range, adjust_yvalues=False)
		xs = np.polyval(pxs, shot_frames)
		ys = np.polyval(pys, shot_frames)
		pixel_matrix = np.array([xs,ys]).T
		return pixel_matrix


def world_shot_position_vectors(frame_info_bundel, shot_frame_range):
	"""
	:param frame_info_bundel: frames_info_bundel object to extract world position vectors from
	:param shot_frame_range: frame range to extract world position vectors from
	:return: return matrix of world position vectors with x,y,z components as columns
	"""
	start_frame, stop_frame = shot_frame_range[0], shot_frame_range[1]
	shot_frames = np.linspace(start_frame, stop_frame, stop_frame-start_frame+1)

	# retrieve normalized polynomial coefficients for x, y and z components within given frame range
	pxs_norm, pys_norm, pzs_norm = find_normalized_ball_regression_formulas(frame_info_bundel, shot_frame_range) #adjusted ys

	# find x,y and z datapoints from normalized polynomial coefficients and given frame range
	xs_norm = np.array(np.polyval(pxs_norm, shot_frames))
	ys_norm = np.array(np.polyval(pys_norm, shot_frames))
	zs_norm = np.array(np.polyval(pzs_norm, shot_frames))

	"""
	# invert y and z values
	neg = lambda t: t*(-1)
	invert_array = np.vectorize(neg)
	ys_norm = invert_array(ys_norm)
	zs_norm = invert_array(zs_norm)

	# scale to balls acutal radius in meters
	ball_radius_meters = 0.12
	xs_meters = np.multiply(xs_norm, ball_radius_meters)
	ys_meters = np.multiply(ys_norm, ball_radius_meters)
	zs_meters = np.multiply(zs_norm, ball_radius_meters)

	# set starting point as origin on all axes
	xs_meters = np.add(xs_meters, -xs_meters[0])
	ys_meters = np.add(ys_meters, -ys_meters[0])
	zs_meters = np.add(zs_meters, -zs_meters[0])
	"""
	### TMP!!!!!
	xs_meters = xs_norm
	ys_meters = ys_norm
	zs_meters = zs_norm


	#return matrix
	return np.array([xs_meters, ys_meters, zs_meters]).T

def get_world_shot_xyzs(frame_info_bundel, shot_frame_range):
	"""
	:param frame_info_bundel: frames_info_bundel object to extract world position vectors from
	:param shot_frame_range: frame range to extract world position vectors from
	:return: return list of numpy arrays of world positions [xs,ys,zs] 
	"""
	world_position_vectors = world_shot_position_vectors(frame_info_bundel, shot_frame_range)
	ball_radius_meters = 0.12
	world_xs_meters = np.multiply(world_position_vectors[:,0],ball_radius_meters)
	world_ys_meters = np.multiply(world_position_vectors[:,1],ball_radius_meters)
	world_zs_meters = np.multiply(world_position_vectors[:,2],ball_radius_meters)
	return [world_xs_meters,world_ys_meters, world_zs_meters]

#source:https://newtonexcelbach.com/2014/03/01/the-angle-between-two-vectors-python-version/
def py_ang(v1, v2, radians=True):
    """ Returns the angle in radians (by defualt) between vectors 'v1' and 'v2'  """
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    angle_radians = np.arctan2(sinang, cosang)
    if radians:
    	return angle_radians
    return math.degrees(angle_radians)

def get_initial_velocity(frame_info_bundel, shot_frame_range):
	"""
	:param frame_info_bundel: frames_info_bundel object to extract initial velocity from
	:param shot_frame_range: frame range to extract initial velocity from
	:return: return initial velocity (m/s)
	"""
	FPS=24	# speed of video
	world_position_vectors = world_shot_position_vectors(frame_info_bundel, shot_frame_range)
	initial_position_vector = world_position_vectors[1]
	initial_velocity_vector = np.multiply(initial_position_vector, FPS)
	initial_velocity = np.linalg.norm(initial_velocity_vector)
	return initial_velocity


def get_launch_angle(frame_info_bundel, shot_frame_range, radians=True):
	"""
	:param frame_info_bundel: frames_info_bundel object to extract launch angle from
	:param shot_frame_range: frame range to extract launch angle from
	:param radians: boolean default True, False will return degrees
	:return: return launch angle defualt radians
	"""
	world_position_vectors = world_shot_position_vectors(frame_info_bundel, shot_frame_range)
	initial_position_vector = world_position_vectors[1]
	initial_x_component_vector = np.array([initial_position_vector[0], 0, 0])
	launch_angle = py_ang(initial_x_component_vector, initial_position_vector,radians)
	return launch_angle


# get error of least squares fit
def get_error(xs,xs_hat):
	assert len(xs) == len(xs_hat)
	squared_error = 0
	for i in range(len(xs)):
		squared_error += abs(xs[i]-xs_hat[i])
	return(math.sqrt(squared_error))

#error of slope fit (degree 2): m=slope, p2, xs, ys
def error_of_slope_fit(m, p2_old, xs ,ys):
	p2 = copy.deepcopy(p2_old)
	p2[0] = m
	y_corrections = []

	# p2 is formula for correction
	c0 = np.polyval(p2,xs[0])
	for x in xs:
		ci = np.polyval(p2, x)
		correction = ci - c0
		y_corrections.append(correction)
	y_corrections = np.array(y_corrections)
	corrected_ys = np.add(y_corrections,ys)

	p2_cy =np.polyfit(xs, corrected_ys, 2)
	p2_corrected_ys = np.polyval(p2_cy, xs)
	print("\nslope: %f" % m)
	print("corrected ys:")
	print(corrected_ys)
	print("new polynomial fit for corrected ys:")
	print(p2_corrected_ys)
	return get_error(p2_corrected_ys, corrected_ys)

#
#
#                                           Main
#
#

if __name__ == '__main__':

	#
	# Initial Evaluation
	#

	shots = [1,2,5,16,18]
	for i in shots: #range(16, 17):

		print ("video %d" % i)

		model_collection_name = "basketball_model_v1" #"person_basketball_model_v1" #

		#input video frames directory paths
		video_frames_dirpath = "/Users/ljbrown/Desktop/StatGeek/object_detection/video_frames/frames_shot_%s" % i

		#output images and video directories for checking
		output_frames_directory = "/Users/ljbrown/Desktop/StatGeek/object_detection/object_detection/%s/output_images/output_frames_shot_%s" % (model_collection_name,i)
		output_video_file = '%s/output_video/shot_%d_detection.mp4' % (model_collection_name,i)

		#image_boolean_bundel and image_info_bundel file paths for quick access
		image_boolean_bundel_filepath = "/Users/ljbrown/Desktop/StatGeek/object_detection/%s/image_evaluator_output/shot_%s_image_boolean_bundel.json" % (model_collection_name,i)
		image_info_bundel_filepath = "/Users/ljbrown/Desktop/StatGeek/object_detection/%s/image_evaluator_output/shot_%s_image_info_bundel.json" % (model_collection_name,i)

		#tensorflow models
		BASKETBALL_MODEL = {'name' : 'basketball_model_v1', 'use_display_name' : False, 'paths' : {'frozen graph': "image_evaluator/models/basketball_model_v1/frozen_inference_graph/frozen_inference_graph.pb", 'labels' : "image_evaluator/models/basketball_model_v1/label_map.pbtxt"}}
		PERSON_MODEL = {'name' : 'ssd_mobilenet_v1_coco_2017_11_17', 'use_display_name' : True, 'paths' : {'frozen graph': "image_evaluator/models/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph/frozen_inference_graph.pb", 'labels' : "image_evaluator/models/ssd_mobilenet_v1_coco_2017_11_17/mscoco_label_map.pbtxt"}}
		BASKETBALL_PERSON_MODEL = {'name' : 'person_basketball_model_v1', 'use_display_name' : False, 'paths' : {'frozen graph': "image_evaluator/models/person_basketball_model_v1/frozen_inference_graph/frozen_inference_graph.pb", 'labels' : "image_evaluator/models/person_basketball_model_v1/label_map.pbtxt"}}
		#bool rule - any basketball or person above an accuracy score of 40.0
		bool_rule = "any('basketball', 40.0) or any('person', 40.0)"

		#
		#     evaluate frame directory
		#               and
		#   save to files for quick access
		#

		#save_image_directory_evaluations(video_frames_dirpath, image_boolean_bundel_filepath, image_info_bundel_filepath, [BASKETBALL_MODEL, PERSON_MODEL], bool_rule)
		#save_image_directory_evaluations(video_frames_dirpath, image_boolean_bundel_filepath, image_info_bundel_filepath, [BASKETBALL_PERSON_MODEL], bool_rule)

		#
		#   load previously evaluated frames
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
		#   Call function for frame cycle
		#

		#output_video_file = 'output_video/shot_%d_detection.mp4' % i
		#frame_cycle(image_info_bundel, input_frame_path_dict, output_frames_directory, output_video_file, pure_boundary_box_frame, apply_history=True)
		#frame_cycle(image_info_bundel, input_frame_path_dict, output_frames_directory, output_video_file, stabalize_to_person_mark_frame)

		#
		#   Intelligently extract all shot frame ranges withing video
		#

		# get shot_frame_ranges
		shot_frame_ranges = find_shot_frame_ranges(image_info_bundel, single_data_point_shots=False)
		

		#
		#   Extract Extrapolated World Shot Datapoints
		#

		
		world_data_matrices = 	[] #[xs_meters, ys_meters, zs_meters, shot_frames]	# get world shot data
		pixel_data_matrices = 	[] #[xs, ys, shot_frames]							# get pixel data

		for sfr in shot_frame_ranges:

			# world data points
			start_frame, stop_frame = sfr[0], sfr[1]
			shot_frames = np.linspace(start_frame, stop_frame, stop_frame-start_frame+1)
			initial_velocity = get_initial_velocity(image_info_bundel, sfr)
			launch_angle_degrees = get_launch_angle(image_info_bundel, sfr, radians=False)
			xs_meters, ys_meters, zs_meters = get_world_shot_xyzs(image_info_bundel, sfr)

			# get shot world matrix (world_shot_position_vectors)
			shot_world_matrix = world_shot_position_vectors(image_info_bundel, sfr)
			world_data_matrices.append(shot_world_matrix)


			#
			# P - get pixel matrix 2D
			#	
			#	* matrix of (x,y) vectors as rows
			#

			#pixel_matrix = find_ball_regression_formulas(image_info_bundel, sfr, adjust_yvalues=False)
			shot_pixel_matrix = pixel_shot_position_vectors(image_info_bundel, sfr, extrapolate=True)
			pixel_data_matrices.append(shot_pixel_matrix)

			# Transform 2D Pixel Matrix to 3D Scaled Camera Matrix

			pxs_norm, pys_norm, pzs_norm = find_normalized_ball_regression_formulas(image_info_bundel, sfr, adjust_yvalues=False, amplify_zslope=False) #false false

			#		2D Point recovery: ( From Camera Coordinates to Pixel Coordinates)
			#
			#	(px,py) = ( (ax*cx)/cz, (ay*cy)/cz )
			#
			#	2D Point to Homogenous coordinates:
			#	
			#	(px,py,1) = ( (ax*cx)/cz, (ay*cy)/cz, 1)
			#
			#
			#  	 | ax  0  0 |   | cx |   | ax*px |
			#    | 0  ay  0 | . | cy | = | ay*py |  divide by cz to recover (2D Point to Homogenous coordinates)
		 	#  	 | 0   0  1 |   | cz |   |   cz  |
		 	#		  (F)		  (c)       (p)*cz
		 	#
		 	#		F . c = p * cz
		 	#		(F . c)/cz = p
		 	#			(c->p)			# c2p
					
			# plug in first frame values from shot_pixel_matrix
			# use this knowledge to find the coeffiecnt ax, ay

			# expand shot pixel matrix to include 1's in z col
			#expand_2d = lambda p, 
			pixel_zs = np.full(len(shot_frames), 1)
			shot_pixel_matrix = np.append(shot_pixel_matrix.T, [pixel_zs], 0).T

			# solve for ax (1/f) with known start normalized vectors
			cx0 = np.polyval(pxs_norm,start_frame)
			cy0 = np.polyval(pys_norm,start_frame)
			cz0 = np.polyval(pzs_norm,start_frame)

			# ax*cx = px*cz --> ax = (px*cz)/cx
			c0 = np.array([cx0,cy0,cz0])
			numerators = np.multiply(shot_pixel_matrix[0], cz0)
			ax, ay, az = np.divide(numerators, c0) 	# az should be equal to 1 here

			# F
			F = np.array([[ax,0,0],[0,ay,0],[0,0,1]])
			F_inv = np.linalg.inv(F)

			# camera coordinates to pixel coordinates
			c2p = lambda c:  np.divide(np.dot(F,c), c[2])

			# checking F and c2p
			#print(c2p(c0))

			#
			#			# p2c
			#	c = (p * cz) . F^(-1)
			#	where cz is found using np.polyval(pzs, frame)
			#

			p2c = lambda p, frame: np.dot(F_inv, np.multiply(p, np.polyval(pzs_norm, frame)))

			# checking p2c
			#print(p2c(shot_pixel_matrix[0], start_frame))
			#print(c0)


			#	Find transformation between shot world matrix and camera matrix
			#		* find translation between shot world matrix and camera matrix
			#			(No find it between normalized shot matrix (True, True) and (False False))
			#	
			#	

			# world shot matrix is a scalled version of this (inverting y and z values and multiplying by 0.12)
			# (.12, -.12, -.12)
			pxs_norm, adjusted_pys_norm, amplified_pzs_norm = find_normalized_ball_regression_formulas(image_info_bundel, sfr) #true true


			# find translation vector t - between camera origin and world origin (location of first frame ball mark)
			wx0 = np.polyval(pxs_norm,start_frame)
			wy0 = np.polyval(adjusted_pys_norm,start_frame)
			wz0 = np.polyval(amplified_pzs_norm,start_frame)
			w0 = np.array([wx0,wy0,wz0])
			t = np.subtract(w0, c0)


			# Find Rotation Matrix

			# expand normalized shot world vecto 0 to include 1 in additional column
			w0 = np.append(w0,[1],0)
			
			# expand camera vector 0 to include 1 in additional column
			c0 = np.append(c0,[1],0)

			# expand t to include -1 in additional coulmn
			t = np.append(t,[-1],0)

			# Create Identity Matrix and replace final coulmn of I with -t
			L = np.eye(4)
			L[:,-1] = -t

			#
			#	c0 = RLw0
			#	R = c0(Lw0)^(-1)
			# 	(Lw0)^(-1) = Lw0_inv
			
			W0 = np.multiply(np.eye(4), w0)
			Lw0_inv = np.linalg.inv(np.dot(L,W0))
			C0 = np.multiply(np.eye(4), c0)
			R = np.dot(C0, Lw0_inv)

			# convert from extened normalized world coordinates to extended camera coordinates
			xnw2xc = lambda xnw: np.dot(R, np.dot(L, xnw))

			# check if it works
			#print(c0)
			#print(xnw2xc(w0))

			#
			#	c = RLw
			#	w = (RL)^(-1)c
			# 
			RL_inv = np.linalg.inv(np.dot(R,L))

			# convert from extened camera coordinates to extended normalized world coordinates
			xc2xnw = lambda xc: np.dot(RL_inv, xc)

			#check 
			#print(w0)
			#print(xc2xnw(c0))

			# convert from normalized world coordinates to camera coordinates
			w2c = lambda w: xnw2xc(np.append(w,[1],0))[:-1]

			# check
			#print(c0[:-1])
			#print(w2c(w0[:-1]))

			# convert from camera coordinates to normalized world coordinates 
			c2w = lambda c: xc2xnw(np.append(c,[1],0))[:-1]

			# check
			#print(w0[:-1])
			#print(c2w(c0[:-1]))


			# convert pixel coordinates to normalized world coordinates
			p0 = pixel_shot_position_vectors(image_info_bundel, sfr, extrapolate=False)[0]
			p2w = lambda p,frame: c2w(p2c(np.append(p,[1],0),frame))

			# check
			#print(w0[:-1])
			#print(p2w(p0,start_frame))

			# convert normalized world coordinates to pixel coordinates
			w2p = lambda w: c2p(w2c(w))[:-1]

			#check 
			#print(p0)
			#print(w2p(w0[:-1]))

			
			#		2D Point recovery
			#			(x,y) = (fX/Z, fY/Z)
			#		2D Point to Homogenous coordinates 
			#			(x,y,1) = (fX/Z, fY/Z, 1)
			#		If Z does not equal zero:
			#			(fX/Z, fY/Z, 1) ~ (fX,fY,Z)
			#		this is useful becuase we can now write the projection of a 3D point
			# 		onto a 2D plane using a (3*4) transformation matrix
			#
			#				  	 | ax  0  0 0 |   | cx |   | ax*px |
			#				     | 0  ay  0 0 | * | cy | = | ay*py |
			#			 	  	 | 0   0  1 0 |   | cz |   |   cz  |
			#								 
			#	(3*4 transformation matrix) (3D Point) - (2D Point Homogeneous Coordinates)




			#
			#	Attempt World Coordinates to Pixel Coordinates
			#		* try camera coordinates orign equal to pixel coordinates orgin 
			#		* world origin is location of ball mark in first frame of shot
			#		* translate: find vector from camera origin to world origin (t)
			#		* rotate: find rotation matrix (R)

			#	cp = <cx, cy, cz, 1>, 
			#	wp = <wx, wy, wz, 1>,
			#	t = <-tx, -ty, -tz, 1>
			#	I = 3*3
			# 	R = 3*3



			#
			#	Mistake ^^ all w's should be nw's
			#

			# convert world to normalaized world coordinates
			# world shot matrix is a scalled version of this (inverting y and z values and multiplying by 0.12)
			# (.12, -.12, -.12) Add first (.12)*([wx0, wy0, wz0]) then multiply

			"""
			nw0 = w0[:-1]
			w0_shift = np.multiply(nw0, np.array([.12,-.12,-.12]))
			print(w0_shift)

			w2nw = lambda w: np.add( np.divide( w, np.array([.12,-.12,-.12]) ), w0_shift)
			nw2w = lambda w: np.divide(w, np.array([.12,-.12,-.12]))


			# check
			print(shot_world_matrix[0])
			print(w2nw(shot_world_matrix[0]))
			print(w0)
			"""


			#
			#	View Plot
			#

			try:
				shot_xs_meters, shot_ys_meters, shot_zs_meters, shot_frames, initial_velocity, launch_angle_degrees = xs_meters, ys_meters, zs_meters, shot_frames, initial_velocity, launch_angle_degrees #world_data[0]

				# try adding original datapoints to plot by transforming them to TMP normalize world coordinates
				original_data = np.array([p2w(p[:-1],frame) for p,frame in zip(shot_pixel_matrix, shot_frames)])
				shot_xs_real,shot_ys_real, shot_zs_real = original_data.T

				ax = plt.axes(projection='3d')
				scat = ax.scatter(shot_xs_real, shot_ys_real, shot_zs_real, c=(1,.45,0), edgecolors=(1,.3,0))
			
				

				ax = plt.axes(projection='3d')
				ax.set_aspect('equal')
				#scat = ax.scatter(shot_xs_meters, shot_ys_meters, shot_zs_meters, c=(1,.45,0), edgecolors=(1,.3,0))
				ax.plot(shot_xs_meters, shot_ys_meters, shot_zs_meters)
				ax.set_xlabel('Xs meters', linespacing=3.2)
				ax.set_ylabel('\tYs meters', linespacing=3.2)
				ax.set_zlabel('\tZs meters', linespacing=3.2)
				ax.yaxis.set_rotate_label(False)
				ax.zaxis.set_rotate_label(False)
				ax.tick_params(direction='out', length=2, width=1, colors='b', labelsize='small')
				# Create cubic bounding box to simulate equal aspect ratio
				max_range = np.array([shot_xs_meters.max()-shot_xs_meters.min(), shot_ys_meters.max()-shot_ys_meters.min(), shot_zs_meters.max()-shot_zs_meters.min()]).max()
				Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(shot_xs_meters.max()+shot_xs_meters.min())
				Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(shot_ys_meters.max()+shot_ys_meters.min())
				Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(shot_zs_meters.max()+shot_zs_meters.min())
				# Comment or uncomment following both lines to test the fake bounding box:
				for xb, yb, zb in zip(Xb, Yb, Zb):
				   ax.plot([xb], [yb], [zb], 'w')

				figure_text = "Video %d\nInitial Velocity %f m/s\nLaunch Angle %f degrees" % (i,initial_velocity, launch_angle_degrees)
				plt.figtext(.25, 0.125, figure_text, style='italic',
		        bbox={'facecolor':'blue', 'alpha':0.5, 'pad':10})
				ax.view_init(elev=140, azim=-90)

				ax.scatter(shot_xs_meters[0], shot_ys_meters[0], shot_zs_meters[0], c='None', s=100,edgecolors='g', linewidths=2)
				plt.grid()


				#fig.canvas.draw()
				plt.show()
			except: 
				pass


