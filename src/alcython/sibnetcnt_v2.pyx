import numpy as np
import time
cimport cython
cimport libcpp
cimport libcpp.queue
cimport libcpp.set
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, preincrement as inc
from collections import defaultdict, OrderedDict

cdef extern from "<algorithm>" namespace "std":
	void std_sort "std::sort" [iter](iter first, iter last)

DTYPE = np.intc
HEIGHT = 256
WIDTH = 256
cdef (pair [int, int])[65536] NUM2PAIRS
cdef int[256][256] PAIRS2NUM
cdef int [8] idy = [0, 1, 1, 1, 0, -1, -1, -1]
cdef int [8] idx = [1, 1, 0, -1, -1, -1, 0, 1]
cdef int [9][8] directions = [[1, 2, 3, 4, 5, 6, 7, 8],
							[4, 5, 6, 0, 0, 0, 0, 0],
							[4, 5, 6, 7, 8, 0, 0, 0],
							[6, 7, 8, 0, 0, 0, 0, 0],
							[1, 2, 6, 7, 8, 0, 0, 0],
							[1, 2, 8, 0, 0, 0, 0, 0],
							[1, 2, 3, 4, 8, 0, 0, 0],
							[2, 3, 4, 0, 0, 0, 0, 0],
							[2, 3, 4, 5, 6, 0, 0, 0]]

cdef int [9] opposite = [0, 5, 6, 7, 8, 1, 2, 3, 4]

#init pairs
for i in range(HEIGHT):
	for j in range(WIDTH):
		PAIRS2NUM[i][j] = i * HEIGHT + j
		NUM2PAIRS[PAIRS2NUM[i][j]] = pair[int, int](i, j)


@cython.boundscheck(False)
@cython.wraparound(False)




def produce_instance_map(int[:, ::1] dual_map, int[:, ::1] left_map, int[:, ::1] top_map, 
							int[:, ::1] top_left_map, int[:, ::1] top_right_map, int num_seed):
	cdef int UG_LEVEL = -1
	cdef int BG_LEVEL = 0
	cdef int FOOD_LEVEL = 1
	cdef int SEED_LEVEL = 2
	cdef int PADDING = 1


	# cdef pixel_id = np.zeros((HEIGHT, WIDTH), dtype=DTYPE)
	# cdef seed_count = np.zeros((HEIGHT, WIDTH), dtype=DTYPE)
	# cdef libcpp.set.set[int] seeds
	# cdef libcpp.set.set[int] tmp_seeds
	cdef int[256][256] pixel_id
	cdef int[256][256] marked_cell
	cdef int[65536] seeds
	cdef int[256][256] from_id  #from which direction 1-8
	cdef int [50] seed_weight
	cdef vector[int] weight_vector
	cdef int begin_id = 0
	cdef int end_id = 0
	cdef int full_expand = 0

	cdef libcpp.queue.queue[int] uncertain_zone



	cdef int run_id = 1
	cdef libcpp.queue.queue[int] q
	# cdef libcpp.set.set[int] s
	cdef vector[int] v
	# cdef libcpp.set.set[int].iterator itr

	# t0 = time.time()

	#collect seeds
	def isInsideBox(int i, int j):
		if i < 0 or i >= HEIGHT: return False
		if j < 0 or j >= WIDTH: return False
		return True

	# print ("start the queue", run_id)
	def collect_at_1(int y, int x, int seed_id):
		if pixel_id[y][x+1] > 0: return seed_id
		if dual_map[y, x+1] == BG_LEVEL: return seed_id
		if left_map[y, x] == 0: 
			# (y, x+1) is located in uncertain zone
			if marked_cell[y][x+1] == 0:
				marked_cell[y][x+1] = 1
				pixel_id[y][x+1] = pixel_id[y][x] #temporary pixel_id
				uncertain_zone.push(PAIRS2NUM[y][x+1])
			return seed_id
		from_id[y][x+1] = 5
		seeds[seed_id] = PAIRS2NUM[y][x+1]
		pixel_id[y][x+1] = pixel_id[y][x]
		return seed_id + 1

	def collect_at_5(int y, int x, int seed_id):
		if pixel_id[y][x-1] > 0: return seed_id
		if dual_map[y, x-1] == BG_LEVEL: return seed_id
		if left_map[y, x-1] == 0:
			if marked_cell[y][x-1] == 0:
				marked_cell[y][x-1] = 1
				pixel_id[y][x-1] = pixel_id[y][x] #temporary pixel_id
				uncertain_zone.push(PAIRS2NUM[y][x-1]) 
			return seed_id
		from_id[y][x-1] = 1
		seeds[seed_id] = PAIRS2NUM[y][x-1]
		pixel_id[y][x-1] = pixel_id[y][x]
		return seed_id + 1

	def collect_at_2(int y, int x, int seed_id):
		if pixel_id[y-1][x+1] > 0: return seed_id
		if dual_map[y-1, x+1] == BG_LEVEL: return seed_id
		if top_right_map[y-1, x+1] == 0:
			if marked_cell[y-1][x+1] == 0:
				marked_cell[y-1][x+1] = 1
				pixel_id[y-1][x+1] = pixel_id[y][x] #temporary pixel_id
				uncertain_zone.push(PAIRS2NUM[y-1][x+1]) 
			return seed_id
		from_id[y-1][x+1] = 6
		seeds[seed_id] = PAIRS2NUM[y-1][x+1]
		pixel_id[y-1][x+1] = pixel_id[y][x]
		return seed_id + 1

	def collect_at_6(int y, int x, int seed_id):
		if pixel_id[y+1][x-1] > 0: return seed_id
		if dual_map[y+1, x-1] == BG_LEVEL: return seed_id
		if top_right_map[y, x] == 0:
			if marked_cell[y+1][x-1] == 0:
				marked_cell[y+1][x-1] = 1
				pixel_id[y+1][x-1] = pixel_id[y][x] #temporary pixel_id
				uncertain_zone.push(PAIRS2NUM[y+1][x-1]) 
			return seed_id
		from_id[y+1][x-1] = 2
		seeds[seed_id] = PAIRS2NUM[y+1][x-1]
		pixel_id[y+1][x-1] = pixel_id[y][x]
		return seed_id + 1

	def collect_at_3(int y, int x, int seed_id):
		if pixel_id[y-1][x] > 0: return seed_id
		if dual_map[y-1, x] == BG_LEVEL: return seed_id
		if top_map[y-1, x] == 0:
			if marked_cell[y-1][x] == 0:
				marked_cell[y-1][x] = 1
				pixel_id[y-1][x] = pixel_id[y][x] #temporary pixel_id
				uncertain_zone.push(PAIRS2NUM[y-1][x]) 
			return seed_id
		from_id[y-1][x] = 7
		seeds[seed_id] = PAIRS2NUM[y-1][x]
		pixel_id[y-1][x] = pixel_id[y][x]
		return seed_id + 1

	def collect_at_7(int y, int x, int seed_id):
		if pixel_id[y+1][x] > 0: return seed_id
		if dual_map[y+1, x] == BG_LEVEL: return seed_id
		if top_map[y, x] == 0:
			if marked_cell[y+1][x] == 0:
				marked_cell[y+1][x] = 1
				pixel_id[y+1][x] = pixel_id[y][x] #temporary pixel_id
				uncertain_zone.push(PAIRS2NUM[y+1][x]) 
			return seed_id
		from_id[y+1][x] = 3
		seeds[seed_id] = PAIRS2NUM[y+1][x]
		pixel_id[y+1][x] = pixel_id[y][x]
		return seed_id + 1

	def collect_at_4(int y, int x, int seed_id):
		if pixel_id[y-1][x-1] > 0: return seed_id
		if dual_map[y-1, x-1] == BG_LEVEL: return seed_id
		if top_left_map[y-1, x-1] == 0:
			if marked_cell[y-1][x-1] == 0:
				marked_cell[y-1][x-1] = 1
				pixel_id[y-1][x-1] = pixel_id[y][x] #temporary pixel_id
				uncertain_zone.push(PAIRS2NUM[y-1][x-1]) 
			return seed_id 

		from_id[y-1][x-1] = 8
		seeds[seed_id] = PAIRS2NUM[y-1][x-1]
		pixel_id[y-1][x-1] = pixel_id[y][x]
		return seed_id + 1

	def collect_at_8(int y, int x, int seed_id):
		if pixel_id[y+1][x+1] > 0: return seed_id
		if dual_map[y+1, x+1] == BG_LEVEL: return seed_id
		if top_left_map[y, x] == 0:
			if marked_cell[y+1][x+1] == 0:
				marked_cell[y+1][x+1] = 1
				pixel_id[y+1][x+1] = pixel_id[y][x] #temporary pixel_id
				uncertain_zone.push(PAIRS2NUM[y+1][x+1]) 
			return seed_id 
		from_id[y+1][x+1] = 4
		seeds[seed_id] = PAIRS2NUM[y+1][x+1]
		pixel_id[y+1][x+1] = pixel_id[y][x]
		return seed_id + 1

	def expand_at(int bi, int bj, int dirs_id, int arr_size):
		cdef int cnt_ = arr_size
		for dir_id in range(arr_size):
			pi = bi + idy[directions[dirs_id][dir_id] - 1]
			pj = bj + idx[directions[dirs_id][dir_id] - 1]

			if dual_map[pi, pj] == SEED_LEVEL:				
				if pixel_id[pi][pj] == BG_LEVEL:
					pixel_id[pi][pj] = UG_LEVEL
					q.push(PAIRS2NUM[pi][pj])
					from_id[pi][pj] = opposite[directions[dirs_id][dir_id]]

				cnt_ -= 1
		# print ("dirs_id", dirs_id, "cnt_", cnt_)
		return cnt_

	# t0 = time.time()
	for i in range(HEIGHT):
		for j in range(WIDTH):
			pixel_id[i][j] = BG_LEVEL

	for i in range(50):
		seed_weight[i] = 0
	
	#seeding
	for i in range(HEIGHT):
		for j in range(WIDTH):
			if dual_map[i, j] == SEED_LEVEL and pixel_id[i][j] == BG_LEVEL:
				pixel_id[i][j] = UG_LEVEL
				v.clear()
				q.push(PAIRS2NUM[i][j])
				while not q.empty():
					p = q.front()
					(bi, bj) = NUM2PAIRS[p]
					v.push_back(PAIRS2NUM[bi][bj])
					q.pop()

					if from_id[bi][bj] == 0: full_expand = expand_at(bi, bj, 0, 8)
					elif from_id[bi][bj] == 1: full_expand = expand_at(bi, bj, 1, 3)
					elif from_id[bi][bj] == 2: full_expand = expand_at(bi, bj, 2, 5)
					elif from_id[bi][bj] == 3: full_expand = expand_at(bi, bj, 3, 3)
					elif from_id[bi][bj] == 4: full_expand = expand_at(bi, bj, 4, 5)
					elif from_id[bi][bj] == 5: full_expand = expand_at(bi, bj, 5, 3)
					elif from_id[bi][bj] == 6: full_expand = expand_at(bi, bj, 6, 5)
					elif from_id[bi][bj] == 7: full_expand = expand_at(bi, bj, 7, 3)
					elif from_id[bi][bj] == 8: full_expand = expand_at(bi, bj, 8, 5)
						
					if full_expand > 0:
						seeds[end_id] = PAIRS2NUM[bi][bj]
						end_id += 1
					
				# if v.size() >= min_num_pixels:
				for vi in range(v.size()):
					(bi, bj) = NUM2PAIRS[v[vi]]
					pixel_id[bi][bj] = run_id

				seed_weight[run_id] = end_id - begin_id
				weight_vector.push_back(end_id - begin_id)
				run_id += 1
				# print ("run_id", run_id)
				begin_id = end_id
				# else:
				# 	end_id = begin_id

	# t1 = time.time()
	#remove light weigth seeds
	# for vi in range(weight_vector.size()):
	# 	print (weight_vector[vi])
	# print ("------")
	std_sort[vector[int].iterator](weight_vector.begin(), weight_vector.end())
	# for vi in range(weight_vector.size()):
	# 	print (weight_vector[vi])
	# print ("------")
	cdef int seed_weight_lowerbound = 32
	if num_seed != -1: seed_weight_lowerbound = max(seed_weight_lowerbound, 
					weight_vector[max(0, weight_vector.size() - num_seed)])
	# print (num_seed, seed_weight_lowerbound)
	for i in range(HEIGHT):
		for j in range(WIDTH):
			if seed_weight[pixel_id[i][j]] < seed_weight_lowerbound:
				pixel_id[i][j] = 0
	# print ("run_id", run_id)
	begin_id = 0


	# t2 = time.time()
	# refresh from_id matrix
	for seed_pixel_id in range(begin_id, end_id):
		(y, x) = NUM2PAIRS[seeds[seed_pixel_id]]
		from_id[y][x] = 0

	while begin_id < end_id:
		# step 1: collect neighbors of seeds[begin_id]
		(y, x) = NUM2PAIRS[seeds[begin_id]]
		if pixel_id[y][x] == 0:
			begin_id += 1
			continue

		if from_id[y][x] == 0: #collect all
			end_id = collect_at_1(y, x, end_id)
			end_id = collect_at_2(y, x, end_id)
			end_id = collect_at_3(y, x, end_id)
			end_id = collect_at_4(y, x, end_id)
			end_id = collect_at_5(y, x, end_id)
			end_id = collect_at_6(y, x, end_id)
			end_id = collect_at_7(y, x, end_id)
			end_id = collect_at_8(y, x, end_id)

		elif from_id[y][x] == 1: 
			end_id = collect_at_4(y, x, end_id)
			end_id = collect_at_5(y, x, end_id)
			end_id = collect_at_6(y, x, end_id)

		elif from_id[y][x] == 2: 
			end_id = collect_at_4(y, x, end_id)
			end_id = collect_at_5(y, x, end_id)
			end_id = collect_at_6(y, x, end_id)
			end_id = collect_at_7(y, x, end_id)
			end_id = collect_at_8(y, x, end_id)

		elif from_id[y][x] == 3:
			end_id = collect_at_6(y, x, end_id)
			end_id = collect_at_7(y, x, end_id)
			end_id = collect_at_8(y, x, end_id)

		elif from_id[y][x] == 4: 
			end_id = collect_at_1(y, x, end_id)
			end_id = collect_at_2(y, x, end_id)
			end_id = collect_at_6(y, x, end_id)
			end_id = collect_at_7(y, x, end_id)
			end_id = collect_at_8(y, x, end_id)

		elif from_id[y][x] == 5: 
			end_id = collect_at_1(y, x, end_id)
			end_id = collect_at_2(y, x, end_id)
			end_id = collect_at_8(y, x, end_id)

		elif from_id[y][x] == 6: 
			end_id = collect_at_1(y, x, end_id)
			end_id = collect_at_2(y, x, end_id)
			end_id = collect_at_3(y, x, end_id)
			end_id = collect_at_4(y, x, end_id)
			end_id = collect_at_8(y, x, end_id)

		elif from_id[y][x] == 7: 
			end_id = collect_at_2(y, x, end_id)
			end_id = collect_at_3(y, x, end_id)
			end_id = collect_at_4(y, x, end_id)

		elif from_id[y][x] == 8: 
			end_id = collect_at_2(y, x, end_id)
			end_id = collect_at_3(y, x, end_id)
			end_id = collect_at_4(y, x, end_id)
			end_id = collect_at_5(y, x, end_id)
			end_id = collect_at_6(y, x, end_id)

		begin_id += 1


	# t3 = time.time()
	#working on uncertain zone
	#from_id become checking grid
	# cdef int cnt = 0
	while not uncertain_zone.empty():
		# cnt += 1
		# print (uncertain_zone.size())
		p = uncertain_zone.front()
		(bi, bj) = NUM2PAIRS[p]
		pixel_id[bi][bj] = pixel_id[bi][bj]
		uncertain_zone.pop()
		for i in range(-1, 2):
			for j in range(-1, 2):
				pi = bi + i
				pj = bj + j
				if dual_map[pi, pj] != BG_LEVEL and marked_cell[pi][pj] == 0 and pixel_id[pi][pj] <= 0:
					pixel_id[pi][pj] = pixel_id[bi][bj]
					uncertain_zone.push(PAIRS2NUM[pi][pj])
					marked_cell[pi][pj] = 1
					# from_id[pi][pj] = 9
	# print (cnt)

	

	#verify assign all
	# t4 = time.time()
	#seeding - filter light weight - expand - uncertain zone
	#2.9 - 0.8 - 4.1 - 2.6
	#4.0 - 0.7 - 4.2 - 1.3
	# print ("{:.4f} - {:.4f} - {:.4f} - {:.4f}".format(t1 - t0, t2 - t1, t3 - t2, t4 - t3))

	return np.asarray(pixel_id)



def collect_seed(int[:, ::1] dual_map, int[:, ::1] left_map, int[:, ::1] top_map, 
							int[:, ::1] top_left_map, int[:, ::1] top_right_map, int num_seed):
	cdef int UG_LEVEL = -1
	cdef int BG_LEVEL = 0
	cdef int FOOD_LEVEL = 1
	cdef int SEED_LEVEL = 2
	cdef int PADDING = 1


	# cdef pixel_id = np.zeros((HEIGHT, WIDTH), dtype=DTYPE)
	# cdef seed_count = np.zeros((HEIGHT, WIDTH), dtype=DTYPE)
	# cdef libcpp.set.set[int] seeds
	# cdef libcpp.set.set[int] tmp_seeds
	cdef int[256][256] pixel_id
	cdef int[256][256] marked_cell
	cdef int[65536] seeds
	cdef int[256][256] from_id  #from which direction 1-8
	cdef int [50] seed_weight
	cdef vector[int] weight_vector
	cdef int begin_id = 0
	cdef int end_id = 0
	cdef int full_expand = 0

	cdef int run_id = 1
	cdef libcpp.queue.queue[int] q
	cdef vector[int] v


	#collect seeds
	def isInsideBox(int i, int j):
		if i < 0 or i >= HEIGHT: return False
		if j < 0 or j >= WIDTH: return False
		return True


	def expand_at(int bi, int bj, int dirs_id, int arr_size):
		cdef int cnt_ = arr_size
		for dir_id in range(arr_size):
			pi = bi + idy[directions[dirs_id][dir_id] - 1]
			pj = bj + idx[directions[dirs_id][dir_id] - 1]
			
			if dual_map[pi, pj] == SEED_LEVEL:				
				if pixel_id[pi][pj] == BG_LEVEL:
					pixel_id[pi][pj] = UG_LEVEL
					q.push(PAIRS2NUM[pi][pj])
					from_id[pi][pj] = opposite[directions[dirs_id][dir_id]]
				cnt_ -= 1
		# print ("dirs_id", dirs_id, "cnt_", cnt_)
		return cnt_

	for i in range(HEIGHT):
		for j in range(WIDTH):
			pixel_id[i][j] = BG_LEVEL

	for i in range(50):
		seed_weight[i] = 0
	
	#seeding
	for i in range(HEIGHT):
		for j in range(WIDTH):
			if dual_map[i, j] == SEED_LEVEL and pixel_id[i][j] == BG_LEVEL:
				pixel_id[i][j] = UG_LEVEL
				v.clear()
				q.push(PAIRS2NUM[i][j])
				while not q.empty():
					p = q.front()
					(bi, bj) = NUM2PAIRS[p]
					v.push_back(PAIRS2NUM[bi][bj])
					q.pop()

					if from_id[bi][bj] == 0: full_expand = expand_at(bi, bj, 0, 8)
					elif from_id[bi][bj] == 1: full_expand = expand_at(bi, bj, 1, 3)
					elif from_id[bi][bj] == 2: full_expand = expand_at(bi, bj, 2, 5)
					elif from_id[bi][bj] == 3: full_expand = expand_at(bi, bj, 3, 3)
					elif from_id[bi][bj] == 4: full_expand = expand_at(bi, bj, 4, 5)
					elif from_id[bi][bj] == 5: full_expand = expand_at(bi, bj, 5, 3)
					elif from_id[bi][bj] == 6: full_expand = expand_at(bi, bj, 6, 5)
					elif from_id[bi][bj] == 7: full_expand = expand_at(bi, bj, 7, 3)
					elif from_id[bi][bj] == 8: full_expand = expand_at(bi, bj, 8, 5)
						
					if full_expand > 0:
						seeds[end_id] = PAIRS2NUM[bi][bj]
						end_id += 1
					
				# if v.size() >= min_num_pixels:
				for vi in range(v.size()):
					(bi, bj) = NUM2PAIRS[v[vi]]
					pixel_id[bi][bj] = run_id

				seed_weight[run_id] = end_id - begin_id
				weight_vector.push_back(end_id - begin_id)
				run_id += 1
				begin_id = end_id

	std_sort[vector[int].iterator](weight_vector.begin(), weight_vector.end())
	cdef int seed_weight_lowerbound = weight_vector[max(0, weight_vector.size() - num_seed)]
	# print (num_seed, seed_weight_lowerbound)
	for i in range(HEIGHT):
		for j in range(WIDTH):
			if seed_weight[pixel_id[i][j]] < seed_weight_lowerbound:
				pixel_id[i][j] = 0

	return np.asarray(pixel_id)
		
