#import os
#import shutil
#import xml.etree.ElementTree as ET


#!/usr/bin/env python

# internal
import logging
import sys 
import os
import glob
import re
import shutil
import xml.etree.ElementTree as ET

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""

take multiple annotation folders with their respective image folders
rename to standard format and put into new annotations folder and image folder


-combine through input of specific directories

-* combine through input master/root directory, output directory name (combined dataset name) name files and dir

	INPUT_DIR/
		PAIR_1_NAME_annotations/
		PAIR_1_NAME_images/ 
		.
		.
		.
		PAIR_n_NAME_annotations/
		PAIR_n_NAME_images/ 	

	COMBINED_DATASET_NAME/
		COMBINED_DATASET_NAME_annotations/
			-COMBINED_DATASET_NAME_i.xml
		COMBINED_DATASET_NAME_images/ 
			-COMBINED_DATASET_NAME_i.JPEG


	command:
				python merge_annotated_datasets.py -i INPUT_DIR -o COMBINED_DATASET_NAME


	# use case 1
	# use case 2 is two combine xml files pointing to same image folder

	
	Notes:

	* works with relative or absolute paths

	# naming format: 
	#					(PAIR_i_NAME_annotations, PAIR_i_NAME_images)
	#					only requirements are trailing: _annotations, _images
	#


	# [todo/notes]
	# 	currently accepts only jpeg images
	#	ensure images are checked if they can be sucessfully opened
	#	clean on images(cap) not annotations lone pairs
"""

#os.path.isabs(my_path)
#source: https://gist.github.com/dideler/2395703

def getopts(argv):
	""" parse commandline aruments with '-' into key value pairs and return dictonary """
	opts = {}  							# Empty dictionary to store key-value pairs.
	while argv: 						# While there are arguments left to parse...
		if argv[0][0] == '-':  			# Found a "-name value" pair.
			opts[argv[0]] = argv[1]  	# Add key and value to the dictionary.
		argv = argv[1:]  				# Reduce the argument list by copying it starting from index 1.
	return opts

# method to get directory path from pair name and templates : "dir_path(pair_names[0], dirname_templates["images"])"
dir_path = lambda input_dir, directory_pair_name, dir_type: os.path.join(input_dir, directory_pair_name + dirname_templates[dir_type])
#ex: print(dir_path("checked_other", "image"))


def get_directory_pair_names(dirname_templates, input_dir):
	""" return list of input pair names """
	regex_template = "([^/]+)%s"
	combined_image_dirpath_string = glob.glob(input_dir + '/*' + dirname_templates['image'])
	combined_annotation_dirpath_string = glob.glob(input_dir + '/*' + dirname_templates['annotation'])
	image_dirnames = re.findall( regex_template % dirname_templates['image'], ''.join(combined_image_dirpath_string) )
	annotation_dirnames = re.findall( regex_template % dirname_templates['annotation'], ''.join(combined_annotation_dirpath_string) )
	return list(set(image_dirnames).intersection(set(annotation_dirnames)))

"""
def get_file_pair_names(directory_pair_name, input_dir):
	# return list of input pair names

	image_filenames_wext = [os.path.basename(path) for path in glob.glob( dir_path(input_dir, directory_pair_name, "image") + "/*" )]
	annotation_filenames_wext = [os.path.basename(path) for path in glob.glob( dir_path(input_dir, directory_pair_name, "annotation") + "/*" )]

	image_filenames = [f.split()[0] for f in image_filenames_wext]
	annotation_filenames = [f.split()[0] for f in annotation_filenames_wext]

	return annotation_filenames_wext
"""

def swap_exentsion(full_filename, new_extension):
	template = "%s.%s" # filename, extension
	filename_base, old_extension = os.path.splitext(full_filename)
	return template % (filename_base, new_extension.strip('.'))

def create_new_annotation_dir(image_directory, annotations_directory_old, annotations_directory_new):

	# if image output directory does not exist, create it
	if not os.path.exists(annotations_directory_new): os.makedirs(annotations_directory_new)

	for image_path in glob.glob(image_directory + "/*"):

		# copy images over with same basename
		try:
			annotations_filename = swap_exentsion(os.path.basename(image_path), 'xml')
			annotations_old_file_path = os.path.join(annotations_directory_old, annotations_filename)
			shutil.copy(annotations_old_file_path, annotations_directory_new)
		except:
			pass

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

def combine_annotation_image_sets(input_image_dirs, input_annotation_dirs, output_image_dir, output_annotation_dir, image_name_template, annotation_name_template, ABSOLUTE_PREFIX):
	
	# if  output directory does not exist, create it
	if not os.path.exists(output_image_dir): os.makedirs(output_image_dir)
	if not os.path.exists(output_annotation_dir): os.makedirs(output_annotation_dir)

	count = 0

	for dir_index in range(len(input_image_dirs)):

		#save annotation xml and corresponding image jpeg to output dirs with new name
		for file in os.listdir(input_image_dirs[dir_index]):

			# add other excepted extensions here
			if file.endswith(".JPEG") or file.endswith(".jpg"):


				identifier_name = os.path.splitext(file)[0]
				input_image_file_fullpath = os.path.join(input_image_dirs[dir_index], file)

	

				#make sure pair exists
				conjurgate_annotation_filename = identifier_name + ".xml"
				conjurgate_annotation_file_full_path = os.path.join(input_annotation_dirs[dir_index], conjurgate_annotation_filename)

				if os.path.isfile(conjurgate_annotation_file_full_path):

					#write to combined image dir
					new_image_filename = image_name_template%count
					new_image_full_path = os.path.join(output_image_dir, new_image_filename)

					if os.path.isfile(input_image_file_fullpath):
						shutil.copy(input_image_file_fullpath, new_image_full_path)  

						#write to combined annotations dir
						new_annotation_full_path = os.path.join(output_annotation_dir, annotation_name_template%count)
						write_new_annotations(conjurgate_annotation_file_full_path, new_annotation_full_path, output_annotation_dir, new_image_filename, new_image_full_path)

						count += 1

						



if __name__ == '__main__':

	# retrieve arguments: -i INPUT_DIR -o COMBINED_DATASET_NAME
	myargs = getopts(sys.argv)
	try:	input_dir, output_dir = myargs['-i'], myargs['-o']
	except:	sys.exit("correct usage: \n\n\tpython merge_annotated_datasets.py -i INPUT_DIR -o COMBINED_DATASET_NAME\n")

	#retrieve input pair names
	dirname_templates = {
		#dir_type, extension
		"image" : "_images", 
		"annotation" : "_annotations"
	}
	new_annotation_directory_extension = "_new"

	# copy to new annotations directory, temporary stage
	pair_names = get_directory_pair_names(dirname_templates, input_dir)

	input_image_dirs = []
	input_annotation_dirs = []
	for name in pair_names:

		image_dir_path = dir_path(input_dir, name, "image")
		annotation_dir_path = dir_path(input_dir, name, "annotation")
		new_annotation_dir_path = annotation_dir_path + new_annotation_directory_extension

		#create_new_annotation_dir(image_dir_path, annotation_dir_path, new_annotation_dir_path)

		input_image_dirs.append(image_dir_path)
		input_annotation_dirs.append(new_annotation_dir_path)
		

	#take multiple annotation folders with their respective image folders
	#rename to standard format and put into new annotations folder and image folder

	#input image annotation sets
	#indexes correspond to paired directories
	#input_image_dirs = ["JPEG_images_set1", "JPEG_images_set2", "JPEG_images_set3"]
	#input_annotation_dirs = ["XML_annotations_set1", "XML_annotations_set2", "XML_annotations_set3"]

	#output image set directory (combiend)
	output_image_dir = "combined_image_set"

	#output annotation set directory (combiend)
	output_annotation_dir = "combined_annotation_set"

	#name templates
	image_name_template = os.path.basename(output_dir) + "_%d.JPEG"
	annotation_name_template = os.path.basename(output_dir) + "_%d.xml"

	ABSOLUTE_PREFIX = output_dir #'/Users/ljbrown/Desktop/image_collecting/gather/'

	combine_annotation_image_sets(input_image_dirs, input_annotation_dirs, output_image_dir, output_annotation_dir, image_name_template, annotation_name_template, ABSOLUTE_PREFIX)


"""
get_glob_file_template = lambda pair_name, dir_type : str( dir_path(pair_name, dir_type) + '/*')
pair_image_filepaths = glob.glob(get_glob_file_template("checked_other", "image"))
print(pair_image_filepaths)
"""

"""
#retrieve file pair names
filename_templates = {
	#dir_type, extension
	"image" : ".JPEG",			#need to make flexible, .jpg
	"annotation" : ".xml"
}
get_glob_file_template = lambda pair_name, dir_type : str( dir_path(pair_name, dir_type) + '/*' + filename_templates[dir_type] )

i_glob_template = get_glob_file_template("checked_other", "image")
print(i_glob_template)
pair_image_filepaths = glob.glob(get_glob_file_template("checked_other", "image"))
print(pair_image_filepaths)

#print (glob_file_template("checked_other", "image"))
"""


# retreive all input child annotation, image directory pair paths

# create new directory with combined dataset name, 
# and child annotation and image directory with _annotation and _image extensions

# loop through input image dirs, for each image (jpeg file) find conjurgate annotation file (xml file)
# if it exists rename with appropriate count and write to new files

"""

