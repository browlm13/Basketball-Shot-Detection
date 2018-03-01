#!/usr/bin/env python

import math

"""
	cartesean coordinates (pixel values) to single number and vice versa
	the snake method

		     1 2 5 10
		   	 4 3 6 11	
		   	 9 8 7 12
		   	 16151413

"""


def to_snake_head(cartesean_pair):
	x, y = cartesean_pair
	if x >= y:
		return (x-1)**2 + y
	else:
		return y**2 - x + 1


def from_snake_head(snake_head):

	#find first and last element values in L
	last_element = math.ceil(math.sqrt(snake_head))**2 # in L
	first_element = math.floor(math.sqrt(snake_head))**2 + 1 #in L
	max_xy = math.floor(math.sqrt(snake_head)) + 1

	x, y = -1,-1

	# if it is not the final element in L (last_element - (first_element-1) = 0)
	length_L = last_element - (first_element -1)
	if length_L == 0:
		x = 1
		y = math.ceil(math.sqrt(snake_head))
	else:
		L_halfway_element_index = int((length_L - 1)/2)	+1 

		if snake_head - first_element < L_halfway_element_index:
			x = max_xy
			y = snake_head - first_element + 1
		else:
			y = max_xy
			x = last_element - snake_head + 1

	return (x,y)

"""
# testing

n = 5
for x in range(1,n):
	for y in range(1,n):
		print("(x,y) = (%d,%d)" % (x,y))
		head = to_snake_head(x,y)
		print("head = %d" % head)
		rx, ry = from_snake_head(head)
		print("(x,y) = (%d,%d) returned\n" % (rx,ry))


for n in range(0,20):
	print(n**2 +1)

for n in range(1,21):
	print(n**2)
"""

