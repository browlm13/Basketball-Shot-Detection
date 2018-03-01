#!/usr/bin/env python

import math
#import numpy as np
import sys
#sys.float_info.min
"""
	cartesean coordinates, time parameter conversion using Archimedean Spiral

	where a is spiral spacing scaler, use small value of a

	t = sqrt(x^2 + y^2)/a
	x = a*t*cos(t)
	y = a*t*sin(t)


	derivation

	#from
	x = a*t*cos(t)
	y = a*t*sin(t)

	# equs (1,2)
	cos(t) = x/(a*t)
	sin(t) = y/(a*t)

	#identity
	cos^2(t) + sin^2(t) = 1

	#substitute equs (1,2) into identity
	(x/(a*t))^2 + (y/(a*t))^2 = 1

	# solve for t
	[(1/a*t)^2 * x^2] + [(1/a*t)^2 * y^2] = 1
	(1/a*t)^2 * [x^2 + y^2] = 1
	x^2 + y^2 = (a*t)^2							# a and t always positive
	sqrt(x^2 + y^2) = a*t
	sqrt(x^2 + y^2)/a = t

	#therfore
	t = sqrt(x^2 + y^2)/a



	---------------
	x(t) = a*t*cos(t)
	y(t) = a*t*sin(t)
	t(x,y) = arctan(y/x)^2


				Fuck that use the snake method

	
		     1 2 5 10
		   	 4 3 6 11	
		   	 9 8 7 12
		   	 16151413




"""

#	x = a*t*cos(t)
#	y = a*t*sin(t)
def archimedean_spiral_to_cartesean(t, a=(1./100000000)):
	x = t*math.cos(t) #a*t*math.cos(t)
	y = t*math.sin(t) #a*t*math.sin(t)
	return (x,y)

#	t = sqrt(x^2 + y^2)/a
#	t(x,y) = arctan(y/x)
def archimedean_spiral_from_cartesean(cartesean_pair, a=(1./100000000)):
	x,y = cartesean_pair
	#t = math.sqrt(x**2 + y**2)/a
	t = math.atan(y/x)**2
	return t


#start Ls' T's

"""
"""

"""
for n in range(0,20):
	print(n**2 +1)

for n in range(1,21):
	print(n**2)
"""

def to_T(x,y):
	if x >= y:
		return (x-1)**2 + y
	else:
		return y**2 - x + 1


def from_T(T):

	#find first and last element values in L
	last_element = math.ceil(math.sqrt(T))**2 # in L
	first_element = math.floor(math.sqrt(T))**2 + 1 #in L
	max_xy = math.floor(math.sqrt(T)) + 1

	x, y = -1,-1

	# if it is not the final element in L (last_element - (first_element-1) = 0)
	length_L = last_element - (first_element -1)
	if length_L == 0:
		x = 1
		y = math.ceil(math.sqrt(T))
	else:
		L_halfway_element_index = int((length_L - 1)/2)	+1 

		if T - first_element < L_halfway_element_index:
			x = max_xy
			y = T - first_element + 1
		else:
			y = max_xy
			x = last_element - T + 1

	return (x,y)

"""
def from_T(T):

	sqrt_T = math.sqrt(T)

	max_area = math.ceil(sqrt_T)**2
	min_area = math.floor(sqrt_T)**2

	#print("T:%d" % T)
	#print("max_area: %d" % max_area)
	#print("min_area: %d\n" % min_area)

	x, y = -1,-1

	# if it is not the final element in L (max_area - min_area = 0)
	length_L = max_area - min_area
	if length_L == 0:
		x = 1
		y = math.ceil(sqrt_T)
	else:
		L_halfway_element_index = int((length_L - 1)/2)	+1 #mshould always be int!! otherwise ur math is wrong
		#print("half way element in L index: %d" % L_halfway_element_index)

		#find first and last element values in L
		first_element = min_area + 1
		last_element = max_area

		#print("first element in L value: %d" % first_element)
		#print("last element in L value: %d" % last_element)

		#x is max if T - first_element < L_halfway_element_index: x is max means x = math.sqrt(min_area) + 1
		#y is max otherwise
		if T - first_element < L_halfway_element_index:
			#print("x is max")
			x = math.sqrt(min_area) + 1 #correct
			y = T - first_element + 1
		else:
			#print("y is max")
			y = math.sqrt(min_area) + 1 # correct
			x = last_element - T + 1

	#print("(x,y) = (%d,%d)\n" % (x,y))
	return (x,y)
"""
n = 5
for x in range(1,n):
	for y in range(1,n):
		print("(x,y) = (%d,%d)" % (x,y))
		T = to_T(x,y)
		print("T = %d" % T)
		rx, ry = from_T(T)
		print("(x,y) = (%d,%d) returned\n" % (rx,ry))


