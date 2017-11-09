"""

resize to 50 * 50
save 4 diffrent angles of photo
save diffrent shades of color

"""

import cv2
import numpy as np
import sys
import os

def load_images(inpath):
    files = [f for f in os.listdir(inpath)]
    images = []
    for f in files:
        print("adding image")
        images.append(cv2.imread(inpath + f))

    return images

def write_images(image_array, outpath, prefix, extension='.jpg'):
    for i,img in enumerate(image_array):
        fname = outpath + str(i) + '_' + prefix + extension
        print(fname)
        cv2.imwrite(fname, img)

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
        grayscale_images.append( cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return grayscale_images

def resize_images(image_array, dimesions=(50,50)):
    """ resize images and return new array of images """
    resized_images = []
    for img in image_array:
        resized_images.append(cv2.resize(img,dimesions))
    return resized_images

dataset_inpath = 'raw_cropped_basketball_images/'
dataset_outpath = 'positive_images/'
prefix = 'basketball'

all_images = load_images(dataset_inpath)
all_images = resize_images(all_images)
all_images = convert_images_to_grayscale(all_images)
all_images = add_rotated_images(all_images)
all_images = add_brightness_adjusted_images(all_images)
write_images(all_images, dataset_outpath, prefix)

"""
test_img = adjust_gamma(all_images[0])
cv2.imshow('detected Edge',test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

