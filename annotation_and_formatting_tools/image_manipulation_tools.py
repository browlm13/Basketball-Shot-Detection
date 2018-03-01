#!/usr/bin/env python

import cv2
import numpy as np

"""
	Image Manipulation Lib
"""
def load_images(inpath):
	files = [f for f in os.listdir(inpath)]
	images = []
	for f in files:
		print("adding image")
		images.append(cv2.imread(inpath + f))

	return images

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape)/2)
  rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
  return result

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def add_brightness_adjusted_images(image_array):
	full_image_array = []
	for img in image_array:
		full_image_array.append(img)
		full_image_array.append(adjust_gamma(img, .90))
		full_image_array.append(adjust_gamma(img, 1.5))
	return full_image_array

def add_rotated_images(image_array):
	full_image_array = []
	for img in image_array:
		full_image_array.append(img)
		full_image_array.append(rotateImage(img, 90))
		full_image_array.append(rotateImage(img, 180))
		full_image_array.append(rotateImage(img, 270))
	return full_image_array

def convert_images_to_grayscale(image_array):
	""" convert images to grayscale and return new array of images """
	grayscale_images = []
	for img in image_array:
		try:
			grayscale_images.append( cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
		except: pass
	return grayscale_images

def resize_images(image_array, dimesions=(50,50)):
	""" resize images and return new array of images """
	resized_images = []
	for img in image_array:
		resized_images.append(cv2.resize(img,dimesions))
	return resized_images


def segmentize(img, segment_width=100, segment_height=100):
	images = []
	# Croping Formula ==> y:h, x:w
	idx, x_axis, x_width,  = 1, 0, segment_width
	y_axis, y_height = 0, segment_height
	#img = cv2.imread(image_path)
	#height, width, dept = img.shape
	width, height = img.shape[:2]
	while y_axis <= height:
		while x_axis <= width:
			crop = img[y_axis:y_height, x_axis:x_width]
			x_axis=x_width
			x_width+=segment_width
			cropped_image_path = "crop/crop%d.png" % idx
			#cv2.imwrite(cropped_image_path, crop)
			#check size
			new_width, new_height = crop.shape[:2]
			if (new_width == segment_width) and (new_height == segment_height):
				images.append(crop)
			idx+=1
		y_axis += segment_height
		y_height += segment_height
		x_axis, x_width = 0, segment_width
	return images

def segment_images(image_array, segment_dim_tuple):
	segment_width, segment_height = segment_dim_tuple
	full_image_array = []
	for img in image_array:
		full_image_array += segmentize(img,segment_width, segment_height)
	return full_image_array