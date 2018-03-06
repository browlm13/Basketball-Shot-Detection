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

def camera_movement_between_frames(frame_image1, frame_image2):
	"""
	:param frame_image1: first numpy array image to compare
	:param frame_image2: second nnumpy array image to compare
	:return: returns boolean True if there is camera movement between frames. Note: ignores object motion between frames
	"""
	pass

def is_video_stable(input_frames_directory):
	"""
	:param input_frames_directory: String of path to directory containing video frame images
	:return: boolean True if there is no camera movement.
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
# columns: frame, x1, x2, y1, y2 ball state / iou bool
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
	
	"""
	# find average distance away
	# find average radii
	np_ball_radii = np.array(ball_radii)
	mean_radius = np_ball_radii.mean()
	start_mean_radius = (mean_radii + ball_radii[0]*2)/3	# estimate of intial radii, weighted tword start radius
	
	# distance estimate from radius
	# get frame path dict
	frame_path_dict = frame_info_bundel_to_frame_path_dict(frame_info_bundel)
	image_height_pixels = load_image_np(frame_path_dict[frames[0]]).shape[0]
	BALL_TRUE_RADIUS = 
	
	#distance_from_radius = lambda r: math.atan(start_mean_radius/image_height_pixels)
	"""

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

def find_normalized_ball_regression_formulas(frame_info_bundel, shot_frame_range, adjust_yvalues=True, return_pzs=False):
	"""
	:param frames_info_bundel: frames_info_bundel object to extract regression formulas from
	:param shot_frame_range: frame range to extract regression formulas from
	:return: normalized regression polynomial coeffiecnts to balls radius in pixels list [pxs,pys]. format pis: [(coeff 0)frame^0, (coeff 1)frame^1, ...] -- (np.polyfit)
	"""
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
	
	"""
	# find average distance away
	# find average radii
	np_ball_radii = np.array(ball_radii)
	mean_radius = np_ball_radii.mean()
	start_mean_radius = (mean_radii + ball_radii[0]*2)/3	# estimate of intial radii, weighted tword start radius
	
	# distance estimate from radius
	# get frame path dict
	frame_path_dict = frame_info_bundel_to_frame_path_dict(frame_info_bundel)
	image_height_pixels = load_image_np(frame_path_dict[frames[0]]).shape[0]
	BALL_TRUE_RADIUS = 
	
	#distance_from_radius = lambda r: math.atan(start_mean_radius/image_height_pixels)
	"""


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

	# normalize xs and ys to change from first datapoint
	regression_xs = np.polyval(pxs, frames)
	regression_ys = np.polyval(pys, frames)
	centered_xs = [x -regression_xs[0] for x in regression_xs]
	centered_ys = [y -regression_ys[0] for y in regression_ys]
	normalized_xs = [x/r for x,r in zip(xs, ball_radii)] 	#normalize to ball radius in pixels
	normalized_ys = [y/r for y,r in zip(ys, ball_radii)] 	#normalize to ball radius in pixels
	normalized_pxs = np.polyfit(frames, normalized_xs, 1)
	normalized_pys = np.polyfit(frames, normalized_ys, 2)

	# return normalized polynomial coefficents
	if return_pzs:
		normalized_zs = pzs_change_coeff
		return [normalized_pxs, normalized_pys, normalized_zs]
	else:
		return [normalized_pxs, normalized_pys]

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

	for i in range(1, 2):

		print ("video %d" % i)

		model_collection_name = "basketball_model_v1" #"person_basketball_model_v1"

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
		#   Extract Extrapolated Shot Datapoints
		#

		# get shot datapoints
		all_shot_data_points_adjusted = [] # [[xs, ys_adjusted, frames], ...]
		all_shot_data_points = [] # [[xs, ys, frames], ...]
		for sfr in shot_frame_ranges:

			# strech/extrapolate datapoints with regression formulas
			shot_frames = np.linspace(sfr[0], sfr[1], sfr[1]-sfr[0])

			# adjusted ys
			pxs, pys_adjusted, pzs = find_normalized_ball_regression_formulas(image_info_bundel, sfr, return_pzs=True) #adjusted ys
			xs = np.polyval(pxs, shot_frames)
			ys_adjusted = np.polyval(pys_adjusted, shot_frames)
			zs = np.polyval(pzs, shot_frames)
			all_shot_data_points_adjusted.append([xs, ys_adjusted, zs, shot_frames])

			"""
			# not adjusted ys
			pxs, pys = find_normalized_ball_regression_formulas(image_info_bundel, sfr, adjust_yvalues=False) #adjusted ys
			xs = np.polyval(pxs, shot_frames)
			ys = np.polyval(pys, shot_frames)
			all_shot_data_points.append([xs, ys, shot_frames])
			"""

		norm_shot_xs, norm_shot_ys_adjusted, norm_shot_zs, shot_frames = all_shot_data_points_adjusted[0]

		# invert y and z values
		neg = lambda t: t*(-1)
		invert_array = np.vectorize(neg)
		norm_shot_ys_adjusted = invert_array(norm_shot_ys_adjusted)
		norm_shot_zs = invert_array(norm_shot_zs)

		# scale to balls acutal radius in meters
		ball_radius_meters = 0.12
		shot_xs_meters = np.array([nx*ball_radius_meters for nx in norm_shot_xs])
		shot_ys_meters = np.array([ny*ball_radius_meters for ny in norm_shot_ys_adjusted])
		shot_zs_meters = np.array([nz*ball_radius_meters for nz in norm_shot_zs])

		ax = plt.axes(projection='3d')
		ax.set_aspect('equal')
		scat = ax.scatter(shot_xs_meters, shot_ys_meters, shot_zs_meters)
		ax.set_xlabel('Xs meters')
		ax.set_ylabel('Ys meters')
		ax.set_zlabel('Zs meters')
		# Create cubic bounding box to simulate equal aspect ratio
		max_range = np.array([shot_xs_meters.max()-shot_xs_meters.min(), shot_ys_meters.max()-shot_ys_meters.min(), shot_zs_meters.max()-shot_zs_meters.min()]).max()
		Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(shot_xs_meters.max()+shot_xs_meters.min())
		Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(shot_ys_meters.max()+shot_ys_meters.min())
		Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(shot_zs_meters.max()+shot_zs_meters.min())
		# Comment or uncomment following both lines to test the fake bounding box:
		for xb, yb, zb in zip(Xb, Yb, Zb):
		   ax.plot([xb], [yb], [zb], 'w')

		plt.grid()
		plt.show()
		"""
		ax.scatter3D(shot_xs_meters, shot_ys_meters, shot_zs_meters)
		ax.set_xlabel('Xs meters')
		ax.set_ylabel('Ys meters')
		ax.set_zlabel('Zs meters')
		plt.show()
		"""

		"""
		ax = plt.axes(projection='3d')
		ax.scatter3D(norm_shot_xs, norm_shot_ys_adjusted, norm_shot_zs)
		ax.set_xlabel('norm Xs')
		ax.set_ylabel('norm Ys')
		ax.set_zlabel('norm Zs')
		plt.show()
		"""
		"""
		# tmp plot
		shot_xs, shot_ys_adjusted, shot_frames = all_shot_data_points_adjusted[0]
		shot_xs, shot_ys, shot_frames = all_shot_data_points[0]

		# invert y values
		neg = lambda t: t*(-1)
		invert_array = np.vectorize(neg)
		shot_ys_adjusted = invert_array(shot_ys_adjusted)
		shot_ys = invert_array(shot_ys)



		#ax = plt.axes(projection='3d')
		#ax.scatter3D(xs, ys, frames, c=ball_state_colors, cmap='Greens')
		#ax.set_xlabel('Xs')
		#ax.set_ylabel('Ys')
		#ax.set_zlabel('frames')

		# plot ajusted
		minimum = min(shot_ys_adjusted + shot_xs + shot_ys)
		maximum = max(shot_ys_adjusted + shot_xs + shot_ys)
		print(shot_ys_adjusted + shot_xs + shot_ys)
		print(shot_xs)
		print(minimum)
		print(maximum)
		plt.ylim(minimum, maximum)
		plt.xlim(minimum, maximum)
		plt.plot(shot_xs, shot_ys_adjusted, c='g')
		plt.xlabel('Xs', fontsize=18)
		plt.ylabel('Ys', fontsize=18)

		#ax.plot3D(shot_xs, shot_ys_adjusted, shot_frames, c='g', linewidth=1)
		

		#plot non adjusted
		plt.ylim(minimum, maximum)
		plt.xlim(minimum, maximum)
		plt.plot(shot_xs, shot_ys, c='r')
		plt.xlabel('Xs', fontsize=18)
		plt.ylabel('Ys', fontsize=18)


		#ax.plot3D(shot_xs, shot_ys, shot_frames, c='r', linewidth=1)

		plt.show()
		"""
