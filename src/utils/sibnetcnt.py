from constants import *
import queue
import numpy as np
from alcython import seed, sibnetcnt_v2
from collections import defaultdict, OrderedDict
from utils import util_color

BG_LEVEL = 0
FOOD_LEVEL = 1
SEED_LEVEL = 2
PADDING = 2

class InstanceSegmentation():
	def __init__(self, sib_masks, vis_path):
		self.dual_map = sib_masks[0]
		self.left_map = sib_masks[1]
		self.top_map = sib_masks[2]
		self.top_left_map = sib_masks[3]
		self.top_right_map = sib_masks[4]
		self.vis_path = vis_path

		(self.height, self.width) = self.dual_map.shape
		self.pixel_id = np.zeros(self.dual_map.shape)
		self.tmp_ids = [[[] for _ in range(self.width)] for _ in range(self.width)]
		self.seeds = []
		self.neighbors = [[[] for _ in range(self.width)] for _ in range(self.width)]

	def process(self):		
		self.collect_seeds()
		self.collect_neighbors()
		self.expand_loop()
		self.produce_instance_map()


	def collect_seeds(self, min_num_pixels=256):
		run_id = 1
		q = queue.Queue()

		for i in range(self.height):
			for j in range(self.width):
				if self.dual_map[i, j] == SEED_LEVEL and self.pixel_id[i][j] == 0:
					buffer = {(i, j): True}
					q.put((i, j))
					while not q.empty():
						p = q.get()
						for id1 in range(-1, 2):
							for id2 in range(-1, 2):
								if self.isInsideBox(p[0] + id1, p[1] + id2) \
										and self.dual_map[p[0] + id1, p[1] + id2] == SEED_LEVEL \
										and (p[0] + id1, p[1] + id2) not in buffer:
									buffer[(p[0] + id1, p[1] + id2)] = True
									q.put((p[0] + id1, p[1] + id2))

					if len(buffer) < min_num_pixels:
						for (bi, bj) in buffer: self.pixel_id[bi, bj] = -1
					else:
						for (bi, bj) in buffer: 
							self.pixel_id[bi, bj] = run_id
							self.seeds.append((bi, bj))
						run_id += 1

	def expand_one(self, seeds):
		buffers = {}
		#step 1: assign a temporary id
		for (y, x) in seeds:
			for (y_, x_) in self.neighbors[y][x]:
				if self.pixel_id[y_, x_] > 0: continue #already assigned id
				self.tmp_ids[y_][x_].append(self.pixel_id[y, x])
				buffers[(y_, x_)] = True
		#step 2: make decision
		pixel_ids = defaultdict(lambda : 0, OrderedDict())
		for (y_, x_) in buffers:
			self.pixel_id[y_, x_] = self.get_pixel_id(y_, x_)
			pixel_ids[self.pixel_id[y_, x_]] += 1

		return buffers

	def expand_loop(self):
		seeds = self.seeds
		loop_id = 0
		while len(seeds) > 0:
			loop_id += 1
			seeds = self.expand_one(seeds)
			# print ("loop {}: seeds = {}".format(loop_id, len(seeds)))

	def collect_neighbors(self):
		for y in range(PADDING, self.height - PADDING):
			for x in range(PADDING, self.width - PADDING):
				if self.dual_map[y, x] == BG_LEVEL: continue # in case of non-food, don't care the neighbor
				if self.left_map[y, x]: self.neighbors[y][x].append((y, x+2))
				if self.left_map[y, x-2]: self.neighbors[y][x-2].append((y, x))
				if self.top_map[y, x]: self.neighbors[y][x].append((y+2, x))
				if self.top_map[y-2, x]: self.neighbors[y-2][x].append((y, x))
				if self.top_left_map[y, x]: self.neighbors[y][x].append((y+2, x+2))
				if self.top_left_map[y-2, x-2]: self.neighbors[y-2][x-2].append((y, x))
				if self.top_right_map[y, x]: self.neighbors[y][x].append((y+2, x-2))
				if self.top_right_map[y-2, x+2]: self.neighbors[y-2][x+2].append((y, x))

	def get_pixel_id(self, y_, x_):
		self.tmp_ids[y_][x_].sort()
		self.tmp_ids[y_][x_].append(-5) #random number
		mode_id, num_occurence, mx_num_occurence = self.tmp_ids[y_][x_][0], 1, 0
		for i in range(1, len(self.tmp_ids[y_][x_])):
			if self.tmp_ids[y_][x_][i] == self.tmp_ids[y_][x_][i-1]:
				num_occurence += 1
			else:
				if num_occurence > mx_num_occurence: mode_id = self.tmp_ids[y_][x_][i-1]
				num_occurence = 1
		return mode_id

	def produce_instance_map(self):
		num_instances = int(np.max(self.pixel_id))
		instance_map = np.zeros((self.height, self.width, 3))
		for instance_id in range(1, num_instances + 1):
			color = np.array(list(util_color.hsv2rgb(instance_id / num_instances, 1)))
			# print (color)
			instance_map[self.pixel_id==instance_id] = color

		instance_map = Image.fromarray(instance_map.astype('uint8'), 'RGB')
		# instance_map = Image.fromarray((self.pixel_id * 64).astype('uint8'), 'L')
		instance_map.save(self.vis_path)

		dual_map = Image.fromarray((self.dual_map * 100).astype('uint8'), 'L')
		dual_map.save(self.vis_path.replace('.png', '_dual.png'))

		left_map = Image.fromarray((self.left_map * 255).astype('uint8'), 'L')
		left_map.save(self.vis_path.replace('.png', '_left.png'))

		top_map = Image.fromarray((self.top_map * 255).astype('uint8'), 'L')
		top_map.save(self.vis_path.replace('.png', '_top.png'))

	def isInsideBox(self, i, j):
		if i < 0 or i >= self.height: return False
		if j < 0 or j >= self.width: return False
		return True

max_num_instances = 16
colors = np.array([list(util_color.hsv2rgb(_id / max_num_instances, 1)) for _id in range(max_num_instances)])


def save_instance_map(sib_masks, vis_path, num_seed):
	colors = np.array(util_color.BBOX_COLORS)

	dual_map = sib_masks[0].astype(np.intc)
	left_map = sib_masks[1].astype(np.intc)
	top_map = sib_masks[2].astype(np.intc)
	top_left_map = sib_masks[3].astype(np.intc)
	top_right_map = sib_masks[4].astype(np.intc)

	if np.max(dual_map) > 0:
		if np.max(dual_map) == 1:
			dual_map = dual_map * 2

		dual_map[:, 0] = 0
		dual_map[:, -1] = 0
		dual_map[0, :] = 0
		dual_map[-1, :] = 0

		tmp = time.time()
		pixel_id = sibnetcnt_v2.produce_instance_map(dual_map, left_map, top_map, top_left_map, top_right_map, num_seed)
		finish_time = time.time()
		
		seed_id = sibnetcnt_v2.collect_seed(dual_map, left_map, top_map, top_left_map, top_right_map, num_seed)
		num_instances = int(np.max(seed_id))

		if num_instances >= 16:
			colors = np.array([list(util_color.hsv2rgb(_id / (num_instances+1), 1)) for _id in range(num_instances+1)])

		seed_map = np.zeros((seed_id.shape[0], seed_id.shape[1], 3))
		for instance_id in range(1, num_instances + 1):
			seed_map[seed_id==instance_id] = colors[instance_id]
	else:
		pixel_id = dual_map
		seed_map = np.zeros((dual_map.shape[0], dual_map.shape[1], 3))
		finish_time, tmp = 0, 0

	seed_map = Image.fromarray(seed_map.astype('uint8'), 'RGB')
	seed_map.save(vis_path.replace('.png', '_seed.png'))

	dual_map = Image.fromarray((dual_map * 100).astype('uint8'), 'L')
	dual_map.save(vis_path.replace('.png', '_dual.png'))

	top_map = Image.fromarray((top_map * 255).astype('uint8'), 'L')
	top_map.save(vis_path.replace('.png', '_top.png'))

	num_instances = int(np.max(pixel_id))
	instance_map = np.zeros((pixel_id.shape[0], pixel_id.shape[1], 3))
	for instance_id in range(1, num_instances + 1):
		instance_map[pixel_id==instance_id] = colors[instance_id]

	instance_map = Image.fromarray(instance_map.astype('uint8'), 'RGB')
	instance_map.save(vis_path)

	return finish_time - tmp


