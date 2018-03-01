"""

	find xml annotation filename missmatches (image file number vs xml file number)


"""

import glob
import os
import xml.etree.ElementTree as ET

def name_without_extension(filename):
	return filename.split('.')[0]

def xml_file_refrence_image(xml_file):
	tree = ET.parse(xml_file)
	root = tree.getroot()
	filename = root.find('filename').text
	return filename

# get all xml files in annotation directory
#annotation_files = [os.path.basename(path) for path in glob.glob("annotations/*")]
annotation_files = glob.glob("annotations/*")

# for each xml file
for xml_file in annotation_files:
	refrence_image_file_woutext = name_without_extension(xml_file_refrence_image(xml_file))
	annotation_file_woutext = name_without_extension(os.path.basename(xml_file))

	if refrence_image_file_woutext != annotation_file_woutext:
		print(xml_file)

