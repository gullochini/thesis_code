import numpy as np
import random
import math

def compute_orders(error_array):

	orders_array = np.zeros(np.shape(error_array))
	size = np.shape(error_array)[0]

	for i in range(size):

		if i == 0 :
			continue
		else:
			for j in range(np.shape(error_array)[1]):

				order = math.log( error_array[i-1,j] / error_array[i,j], 2)
				orders_array[i,j] += order

	return orders_array

def compute_one_order(error_array):

	orders_array = np.zeros(np.shape(error_array))
	size = np.shape(error_array)[0]

	for i in range(size):

		if i == 0 :
			continue
		else:

			order = math.log( error_array[i-1] / error_array[i], 2)
			orders_array[i] += order

	return orders_array

def compute_EOC_orders(incr_list):

	orders_array = np.zeros(len(incr_list))

	for i, el in enumerate(incr_list):

		if i < 1 or i == len(incr_list) - 1 :
			continue
		else:
			order = math.log( incr_list[i+1] / el, 2) / math.log( el / incr_list[i-1], 2) 
			orders_array[i] += order

	return orders_array


if __name__ == '__main__':

	pass