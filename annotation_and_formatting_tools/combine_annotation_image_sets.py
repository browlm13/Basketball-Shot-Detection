import os
import shutil


import xml.etree.ElementTree as ET
"""

take multiple annotation folders with their respective image folders
rename to standard format and put into new annotations folder and image folder
"""

#input image annotation sets
#indexes correspond to paired directories
input_image_dirs = ["JPEG_images_set1", "JPEG_images_set2", "JPEG_images_set3"]
input_annotation_dirs = ["XML_annotations_set1", "XML_annotations_set2", "XML_annotations_set3"]

#output image set directory (combiend)
output_image_dir = "combined_image_set"

#output annotation set directory (combiend)
output_annotation_dir = "combined_annotation_set"

#name templates
image_name_template = "basketball_%d.JPEG"
annotation_name_template = "basketball_%d.xml"

ABSOLUTE_PREFIX = '/Users/ljbrown/Desktop/image_collecting/gather/' #'/Users/ljbrown/Desktop/object_detection/image_annotation_sets/'

#function needed to change xml file of new annotation
def write_new_annotations(old_annotations_full_path, new_annotation_full_path, new_folder, new_image_filename, new_image_full_path):
		xml = ET.parse(old_annotations_full_path)
		root_element = xml.getroot()

		for folder_tag in root_element.iter(tag='folder'):
			folder_tag.text = new_folder

		for filename_tag in root_element.iter(tag='filename'):
			filename_tag.text = new_image_filename

		for path_tag in root_element.iter(tag='path'):
			path_tag.text = absolute_new_annotation_full_path = ABSOLUTE_PREFIX + new_image_full_path

		xml.write(new_annotation_full_path)


count = 0

for dir_index in range(len(input_image_dirs)):

	#save annotation xml and corresponding image jpeg to output dirs with new name
	for file in os.listdir(input_image_dirs[dir_index]):
		print(file)
		if file.endswith(".JPEG"):

			identifier_name = os.path.splitext(file)[0]
			input_image_file_fullpath = os.path.join(input_image_dirs[dir_index], file)

			#make sure pair exists
			conjurgate_annotation_filename = identifier_name + ".xml"
			conjurgate_annotation_file_full_path = os.path.join(input_annotation_dirs[dir_index], conjurgate_annotation_filename)
			if os.path.isfile(conjurgate_annotation_file_full_path):

				#write to combined image dir
				new_image_filename = image_name_template%count
				new_image_full_path = os.path.join(output_image_dir, new_image_filename)
				shutil.copy(input_image_file_fullpath, new_image_full_path)  

				#write to combined annotations dir
				new_annotation_full_path = os.path.join(output_annotation_dir, annotation_name_template%count)
				write_new_annotations(conjurgate_annotation_file_full_path, new_annotation_full_path, output_annotation_dir, new_image_filename, new_image_full_path)

				count += 1



