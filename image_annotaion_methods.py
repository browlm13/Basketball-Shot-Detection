#!/usr/bin/env python
#python 3

# internal
import logging
import os

# external
import cv2
import numpy as np
import glob
import pandas as pd
import xml.etree.ElementTree as ET

"""
#structure
image_data = {
	'image'= ,
	'path',
	'folder',
	'filename',
	'database',		#raw image foldername
	'class',		#name
	'width',
	'height',
	'xmin',
	'ymin',
	'xmax',
	'ymax',
	'depth',
	'segmented',
	'trucated',
	'difficult',
	'pose'			#unspecified
}

"""

#
# Constants
#

image_filename_template = "%s_%d.JPEG" # class_name, index_id

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def swap_exentsion(full_filename, new_extension):
	template = "%s.%s" # filename, extension
	filename_base, old_extension = os.path.splitext(full_filename)
	return template % (filename_base, new_extension.strip('.'))

def generate_new_filename(output_directory_path, image_data, new_extension):
	new_filename = swap_exentsion(image_data['filename'], new_extension)
	full_path = os.path.join(output_directory_path, new_filename)
	return full_path

#
# Reading And Writing Images
#

def load_images(dir_path):
	""" Read cv2 images from directory path parameter and return array. """

	# ensure directory exists
	assert os.path.exists(dir_path)
	assert os.path.isdir(dir_path)

	#load images files into array
	files = [f for f in os.listdir(dir_path)]
	images = []
	for f in files:
		file_path = os.path.join(dir_path, f)
		try:	
			img = cv2.imread(file_path)
			assert img is not None
			images.append(img)
		except: logger.info('unable to load %s', file_path)

	# ensure images were loaded into array
	assert len(images) > 0

	logger.info('loaded %d images', len(images))
	return images

def write_images(images, file_paths):
	""" write cv2 images to file paths at coresponding indexs """

	assert len(images) == len(file_paths)
	logger.info('Writing %d images', len(images))

	for img, fpath in zip(images, file_paths):
		directory_path = os.path.dirname(fpath)
		if not os.path.exists(directory_path):
			os.makedirs(directory_path)
		cv2.imwrite(fpath, img)

	logger.info('Finished writing %d images', len(images))


def write_annotated_image(image_data):
	""" write cv2 image to file path specified in dictonary """

	# if directorydoes not exist, create it
	if not os.path.exists(image_data['base_path']):
		os.makedirs(image_data['base_path'])

	cv2.imwrite(image_data['full_path'], image_data['image'])

	logger.info('Wrote %s image to file', image_data['filename'])


def write_xml_file(image_data, outpath):
	
	# if directorydoes not exist, create it
	if not os.path.exists(outpath):
		os.makedirs(outpath)

	xml_string = generate_xml_string(image_data)
	xml_filename = generate_new_filename(outpath, image_data, 'xml')

	with open(xml_filename, "w") as f:
		f.write(xml_string)

def xml_to_csv(input_xml_directory_path, output_csv_fullpath):
	""" source: https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py """
	xml_list = []
	for xml_file in glob.glob(input_xml_directory_path + '/*.xml'):
		tree = ET.parse(xml_file)
		root = tree.getroot()
		for member in root.findall('object'):
			value = (root.find('filename').text,
					 int(root.find('size')[0].text),
					 int(root.find('size')[1].text),
					 member[0].text,
					 int(member[4][0].text),
					 int(member[4][1].text),
					 int(member[4][2].text),
					 int(member[4][3].text)
					 )
			xml_list.append(value)
	column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
	xml_df = pd.DataFrame(xml_list, columns=column_name)
	xml_df.to_csv(output_csv_fullpath, index=None)


def make_image_data_dict(image, path, folder, filename, database, class_name):
	""" return image dictonary """

	height, width, depth = image.shape
	image_data = {
		'image' : image,
		'full_path' : path + '/' + filename,	#'path' when xml tag
		'base_path' : path,
		'folder' : folder,
		'filename' : filename,
		'database' : database,					#input directory name
		'class_name' : class_name, 				#'class' when xml tag
		'width' : width,
		'height' : height,
		'depth' : depth,
		'xmin' : 0,
		'ymin' : 0,
		'xmax' : width,
		'ymax' : height,
		'segmented' : 0,
		'pose' : 'Unspecified',
		'truncated' : 0,
		'difficult' : 0,
	}
	return image_data

def generate_xml_string(image_data):

	# create XML 
	annotation_tag = ET.Element('annotation')

	folder_tag = ET.SubElement(annotation_tag, 'folder')
	folder_tag.text = image_data['folder']

	filename_tag = ET.SubElement(annotation_tag, 'filename')
	filename_tag.text = image_data['filename']

	path_tag = ET.SubElement(annotation_tag, 'path')
	path_tag.text = image_data['full_path']

	source_tag = ET.SubElement(annotation_tag, 'source')
	database_tag = ET.SubElement(source_tag, 'database')
	database_tag.text = image_data['database']

	size_tag = ET.SubElement(annotation_tag, 'size')
	width_tag = ET.SubElement(size_tag, 'width')
	width_tag.text = str(image_data['width'])
	height_tag = ET.SubElement(size_tag, 'height')
	height_tag.text = str(image_data['height'])
	depth_tag = ET.SubElement(size_tag, 'depth')
	depth_tag.text = str(image_data['depth'])

	segmented_tag = ET.SubElement(annotation_tag, 'segmented')
	segmented_tag.text = str(0)

	object_tag = ET.SubElement(annotation_tag, 'object')
	name_tag = ET.SubElement(object_tag, 'name')
	name_tag.text = image_data['class_name']
	pose_tag = ET.SubElement(object_tag, 'pose')
	pose_tag.text = image_data['pose']
	truncated_tag = ET.SubElement(object_tag, 'truncated')
	truncated_tag.text = str(image_data['truncated'])
	difficult_tag = ET.SubElement(object_tag, 'difficult')
	difficult_tag.text = str(image_data['difficult'])
	bndbox_tag = ET.SubElement(object_tag, 'bndbox')
	xmin_tag = ET.SubElement(bndbox_tag, 'xmin')
	xmin_tag.text = str(image_data['xmin'])
	ymin_tag = ET.SubElement(bndbox_tag, 'ymin')
	ymin_tag.text = str(image_data['ymin'])
	xmax_tag = ET.SubElement(bndbox_tag, 'xmax')
	xmax_tag.text = str(image_data['xmax'])
	ymax_tag = ET.SubElement(bndbox_tag, 'ymax')
	ymax_tag.text = str(image_data['ymax'])

	return ET.tostring(annotation_tag).decode('utf-8')


def load_annotated_images(input_directory_path, output_directory_path, class_name):

	global image_filename_template

	# ensure directory exists
	assert os.path.exists(input_directory_path)
	assert os.path.isdir(input_directory_path)

	# base directory name from datasetpath / input_directory_path
	database = os.path.basename(os.path.normpath(input_directory_path))

	# base directoy name from output_directory_path, 'folder' in image_data
	folder = os.path.basename(os.path.normpath(output_directory_path))

	# load images
	images = load_images(input_directory_path)

	# annotate images and return list of image data dictonaries
	annotated_images = []
	for index_id, img in enumerate(images):
		filename = image_filename_template % (class_name, index_id)
		image_data = make_image_data_dict(img, output_directory_path, folder, filename, database, class_name)
		annotated_images.append(image_data)


	return annotated_images


def annotate_dataset(input_dataset_path, class_name, output_image_directory_path, output_xml_directory_path, output_csv_full_filepath = None):

	for image_data in load_annotated_images(input_dataset_path, output_image_directory_path, class_name):
		write_annotated_image(image_data)
		write_xml_file(image_data, output_xml_directory_path)

	if output_csv_full_filepath is not None:
		xml_to_csv(output_xml_directory_path, output_csv_full_filepath)



#
#	Testing
#

#	Testing Variables
input_dataset_path = '/Users/ljbrown/Desktop/object_detection/generate_annotated_images/raw_image_sets/positive_basketballs'
image_output_dataset_path = '/Users/ljbrown/Desktop/object_detection/images'
xml_output_directory_path = '/Users/ljbrown/Desktop/object_detection/annotations'
#csv_output_file_fullpath = '/Users/ljbrown/Desktop/data_and_models/data/annotated_image_sets/positive_basketballs/positive_basketballs.csv'
class_name = 'basketball'

annotate_dataset(input_dataset_path, class_name, image_output_dataset_path, xml_output_directory_path)

