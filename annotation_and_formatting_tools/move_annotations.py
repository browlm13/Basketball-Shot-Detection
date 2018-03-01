#python 3

"""
move all annotations that correspon to an image to another directory

"""
import glob
import shutil
import os


"""

- eleminate lone anotations and merge directories (leave old image and annotation directories intact)

"""

image_directory = "/Users/ljbrown/Desktop/StatGeek/object_detection/basketball_dataset_v3/edited_images"
annotations_directory_old = "/Users/ljbrown/Desktop/StatGeek/object_detection/basketball_dataset_v3/edited_annotations"
annotations_directory_new = "/Users/ljbrown/Desktop/StatGeek/object_detection/basketball_dataset_v3/new_edited_annotations"

def swap_exentsion(full_filename, new_extension):
	template = "%s.%s" # filename, extension
	filename_base, old_extension = os.path.splitext(full_filename)
	return template % (filename_base, new_extension.strip('.'))

# if image output directory does not exist, create it
if not os.path.exists(annotations_directory_new): os.makedirs(annotations_directory_new)

for image_path in glob.glob(image_directory + "/*"):

	# copy images over with same basename
	try:
		annotations_filename = swap_exentsion(os.path.basename(image_path), 'xml')
		annotations_old_file_path = os.path.join(annotations_directory_old, annotations_filename)
		print(annotations_old_file_path)
		print(annotations_filename)
		shutil.copy(annotations_old_file_path, annotations_directory_new)
	except:
		pass
