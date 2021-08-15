import numpy as np
cimport cython
cimport libcpp
cimport libcpp.queue
cimport libcpp.set
from cython.operator cimport dereference as deref, preincrement as inc

DTYPE = np.intc

@cython.boundscheck(False)
@cython.wraparound(False)




def count_seed(int[:, ::1] dual_map, int seed_level, int min_num_pixels):
	h = dual_map.shape[0]
	w = dual_map.shape[1]

	def isInsideBox(i, j):
		if i < 0 or i >= h: return False
		if j < 0 or j >= w: return False
		return True

	pixel_id = np.full((h, w), -1, dtype=DTYPE)
	cdef int run_id = 0
	cdef libcpp.queue.queue[(int, int)] q
	cdef libcpp.set.set[int] s
	cdef libcpp.set.set[int].iterator itr

	# print ("start the queue", run_id)
	for i in range(h):
		for j in range(w):
			if dual_map[i, j] == seed_level and pixel_id[i][j] == -1:
				# print (s.size())
				s.clear()
				s.insert(i * h + j)
				q.push((i, j))
				while not q.empty():
					p = q.front()
					q.pop()
					for id1 in range(-1, 2):
						for id2 in range(-1, 2):
							if isInsideBox(p[0] + id1, p[1] + id2) \
									and dual_map[p[0] + id1, p[1] + id2] == seed_level \
									and s.find((p[0] + id1) * h + (p[1] + id2)) == s.end():
								s.insert((p[0] + id1) * h + (p[1] + id2))
								q.push((p[0] + id1, p[1] + id2))
								# print ('update queue', p[0] + id1, p[1] + id2)

				# print ("queue ends")
				itr = s.begin()
				if s.size() < min_num_pixels:
					while itr != s.end():
						bij = deref(itr)
						inc(itr)
						bj = int(bij % h)
						bi = int((bij - bj) / h)
						pixel_id[bi, bj] = -2
				else:
					while itr != s.end():
						bij = deref(itr)
						inc(itr)
						bj = int(bij % h)
						bi = int((bij - bj) / h)
						pixel_id[bi, bj] = run_id
					run_id += 1

				# print (run_id, s.size())
	return run_id