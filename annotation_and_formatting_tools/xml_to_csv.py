#!/usr/bin/env python
#python 3

# internal
import logging
import os

# external
#import cv2
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
np.random.seed(1)


# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""

    source : https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py

	1.)
		from object_detection/images/ and from object_detection/annotations
		take 90% xml image pairs and copy into train folder, put the remaining 10% into the test folder
		(object_detection/images/test, object_detection/images/train)

	2.)
		generate .csv files from xml files and place into corresponding test or train folders into
			-object_detection/data/test/test.csv
			-object_detection/data/train/train.csv
"""


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
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
    return xml_df


def main():

    #
    #settings
    #
    
    class_name = "basketball"
    filename = class_name + '.csv'
    full_csv_filepath = "basketball_dataset_v2/csv_records/" + filename
    train_csv_filepath = "basketball_dataset_v2/csv_records/train_labels.csv"
    test_csv_filepath = "basketball_dataset_v2/csv_records/test_labels.csv"
    ratio_train_images = .9  #90%

    # generate csv file for entire image set
    image_path = os.path.join(os.getcwd(), 'basketball_dataset_v2/annotations')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv(full_csv_filepath, index=None)
    print('Successfully converted xml to csv.')

    """
    class_name = "basketball"
    filename = class_name + '.csv'
    full_csv_filepath = "data/" + filename
    train_csv_filepath = "data/train_labels.csv"
    test_csv_filepath = "data/test_labels.csv"
    ratio_train_images = .9  #90%


    # generate csv file for entire image set
    image_path = os.path.join(os.getcwd(), 'annotations')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv(full_csv_filepath, index=None)
    print('Successfully converted xml to csv.')
    """

    # split into test and train csvs
    grouped = xml_df.groupby('filename')
    grouped_list = [grouped.get_group(line) for line in grouped.groups]
    total_num_files = len(grouped_list)
    num_train_files = int(ratio_train_images * total_num_files)
    train_index = np.random.choice(len(grouped_list), size=num_train_files, replace=False)
    test_index = np.setdiff1d(list(range(total_num_files)), train_index)
    train = pd.concat([grouped_list[i] for i in train_index])
    test = pd.concat([grouped_list[i] for i in test_index])
    train.to_csv(train_csv_filepath, index=None)
    test.to_csv(test_csv_filepath, index=None)


main()