#!/usr/bin/env python

import logging
import numpy as np

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""
				Perspective Transformations on Image Matrices

				definitions:

						* Top Left Origin Coordinate System With positive y axis flipped - S (standard)
						* Image Center Origin Coordinate System - C (centered)

				use homogeneous coordinate systems to translate Matricies

				Transformation_S2C - T_sc

				|	1 	 0	 -img_width/2  |
				| 	0	-1	 img_height/2  | 
				|	0	 0		  1		   |

				Transformation_C2S - T_cs

				|	1 	 0	 img_width/2   |
				| 	0	-1	 img_height/2  | 
				|	0	 0		  1		   |


				T_sc use:

				|	1 	 0	 -img_width/2  | | s_x |     | c_x |
				| 	0	-1	 img_height/2  | | s_y |  =  | c_y |
				|	0	 0		  1		   | |  1  |     |  1  |


 precieved radius
 - pr
 actual radius
 -ar

 
"""

# testing
img_width = 4
img_height = 4
t_x, t_y = (img_width/2), (img_height/2)
T_sc = np.array([[1,0,-t_x],[0,-1,t_y],[0,0,1]])
T_cs = np.array([[1,0,t_x],[0,-1,t_y],[0,0,1]])


def to_homogeneous(np_array):
	# add extra dimension slice of 0's, make final elemnt 1
	lshape = list(np_array.shape)
	lshape[0]=1
	new_row = np.zeros(tuple(lshape), dtype=int)
	np_array_add0s = np.concatenate((np_array, new_row), axis=0)
	indexer = np.array(np_array_add0s.shape)
	indexer = tuple(np.subtract(indexer, 1))
	np_array_add0s[indexer] = 1
	return np_array_add0s

def from_homogeneous(np_array):
	# remove extra dimension slice
	return np.delete(np_array, (-1), axis=0)


# testing

# transofrorm coordinates
s_v = np.array([0,4])
print(s_v)
s_vh = to_homogeneous(s_v)
c_vh = T_sc.dot(s_vh)
c_v = from_homogeneous(c_vh)
print(c_v)
c_vh = to_homogeneous(c_v)
s_vh = T_cs.dot(c_vh)
print(from_homogeneous(s_vh))




