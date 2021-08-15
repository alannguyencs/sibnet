from constants import *


MAX_COUNTING = 20


class EvalCounting():
	def __init__(self):		
		self.num_images = [0 for _ in range(MAX_COUNTING)]
		self.counting_error = [0 for _ in range(MAX_COUNTING)]
		self.counting_correct = [0 for _ in range(MAX_COUNTING)]
		self.max_counting = -1

		self.mae = 0
		self.accuracy = 0

		self.num_images_all = 0
		self.mae_all = 0
		self.accuracy_all = 0

		self.result_detail = OrderedDict()
		self.detail_result_file = None

	
	def update_batch_gt_pred_counting(self, gt_counting, pred_counting):
		batch_size_ = gt_counting.size(0)

		for batch_id in range(batch_size_):
			gt_cnt = int(gt_counting[batch_id][0].item())
			pred_cnt = pred_counting[batch_id][0].item()
			pred_cnt = round(pred_cnt, 0)
			abs_error = abs(pred_cnt - gt_cnt)
			if gt_cnt >= MAX_COUNTING: continue

			self.num_images[gt_cnt] += 1
			self.counting_error[gt_cnt] += abs_error
			self.counting_correct[gt_cnt] += int(abs_error <= 0.5)

			self.num_images_all += 1
			self.mae_all += abs_error
			self.accuracy_all += int(abs_error <= 0.5)

	def summary(self):
		for counting_id in range(MAX_COUNTING):
			if self.num_images[counting_id] > 0:
				self.counting_error[counting_id] /= self.num_images[counting_id]
				self.counting_correct[counting_id] /= self.num_images[counting_id]
				self.max_counting = max(self.max_counting, counting_id)

		self.mae = sum(self.counting_error[1:self.max_counting+1]) / self.max_counting
		self.accuracy = sum(self.counting_correct[1:self.max_counting+1]) / self.max_counting

		self.mae_all /= max(self.num_images_all, 1)
		self.accuracy_all /= max(self.num_images_all, 1)

	def get_summarization(self):
		self.summary()
		return "counting error = {:.4f} | accuracy = {:.4f}".format(self.mae, self.accuracy)

	def show_detail_counting_result(self):
		for counting_id in range(1, self.max_counting + 1):
			print ("gt counting = {}: mae = {:.4f}, accuracy = {:.4f}".format(
				counting_id, self.counting_error[counting_id], self.counting_correct[counting_id]))

		print ("Overall: MAE = {:.4f}, Accuracy = {:.4f}".format(self.mae_all, self.accuracy_all))



	



class DetailCounting():
	def __init__(self, ckpt_path, loader):		
		self.detail_result_file = None
		self.init_detail_result(ckpt_path, loader)

	def write_batch_result(self, image_paths, pred_counting):
		batch_size_ = pred_counting.size(0)

		for batch_id in range(batch_size_):
			pred_cnt = pred_counting[batch_id][0].item()
			self.detail_result_file.write("{}: {}\n".format(image_paths[batch_id], pred_cnt))

	def init_detail_result(self, ckpt_path, loader):
		model_name = ckpt_path.split(CKPT_PATH)[-1][:-5]
		detail_result_path = "{}{}_on_{}.txt".format(RESULT_PATH, model_name, loader.name)
		self.detail_result_file = open(detail_result_path, 'w')
		print ("result is saved in {}".format(detail_result_path))





