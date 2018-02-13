import os
import shutil

import xml.etree.ElementTree as ET
from xml.dom import minidom

"""

take multiple annotation folders decribing all the same image folder and comine object section of annotations
put into new annotations folder combined annoations set
"""

#input image annotation sets
input_annotation_dirs = ["test_annotations_1", "test_annotations_2"]

#output annotation set directory (combiend)
output_annotation_dir = "combined_annotation_set"

#ABSOLUTE_PREFIX = '/Users/ljbrown/Desktop/object_detection/image_annotation_sets/'

#boolean does file exist in directory 
def does_file_exist_in_dir(dirpath, filename):
	return os.path.isfile(os.path.join(dirpath, filename))

# combine diffrent annotation xml files discribing the same image by including all detected objects and return xml string
def combine_annotation_files(disjoint_annfile_paths):

	xmls = []
	for p in disjoint_annfile_paths:
		xmls.append(ET.parse(p))

	#return xml unchanged
	if len(xmls) == 1:
		root_element = xml.getroot()
		#dom = minidom.parseString(ET.tostring(root_element).decode('utf-8'))
		#return dom.toprettyxml(indent='\t')
		return ET.tostring(root_element).decode('utf-8')

	#otherwise add all unique object tags to giant xml file
	combined_xmls_root = xmls[0].getroot()
	for xml in xmls[1:]:
		i_root = xml.getroot()
		for object_tag in i_root.iter(tag='object'):
			combined_xmls_root.append(object_tag)

	#dom = minidom.parseString(ET.tostring(combined_xmls_root).decode('utf-8'))
	#return dom.toprettyxml(indent='\t')
	return ET.tostring(combined_xmls_root).decode('utf-8')

def write_xml_file(output_dirpath, xml_filename, xml_string):
  
	# if directorydoes not exist, create it
	if not os.path.exists(output_dirpath):
		os.makedirs(output_dirpath)

	outpath = os.path.join(output_dirpath, xml_filename)
	with open(outpath, "w") as f:
		f.write(xml_string)

#
# find all unqiue xml file names used in all sets of folders
#

all_annotation_filenames = []
for input_dir in input_annotation_dirs:
	for file in os.listdir(input_dir):
		if file.endswith(".xml"):
			all_annotation_filenames.append(file)
unique_annotation_filenames = set(all_annotation_filenames)

#
# load seperate annotation xmls for each filename
#

for ann_filename in unique_annotation_filenames:

	# loacte all filepaths for annotation files decribing this image
	disjoint_annfile_paths = []
	for input_dir in input_annotation_dirs:
		ann_path = os.path.join(input_dir, ann_filename)
		if os.path.isfile(ann_path):
			disjoint_annfile_paths.append(ann_path)

	assert len(disjoint_annfile_paths) > 0

	# combine dijoint annotation files
	xml_string = combine_annotation_files(disjoint_annfile_paths)

	#write to new folder
	write_xml_file(output_annotation_dir, ann_filename, xml_string)




"""

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
"""



