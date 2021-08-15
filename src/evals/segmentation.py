from constants import *
from collections import defaultdict, OrderedDict

class EvalBinarySegmentation():
	def __init__(self):
		self.confusion_matrix = np.zeros((2, 2), dtype=np.float64)
		self.num_categories = 2   #non-food and food
		self.num_gt = np.zeros(2, dtype=np.float64)     #non-food, food

		self.cate_names = ["background", "foreground"]
		self.accuracy = np.zeros(2, dtype=np.float64)
		self.IoU = np.zeros(2, dtype=np.float64)

		self.pixel_accuracy = 0
		self.mean_accuracy = 0
		self.mean_IoU = 0


	def update_batch_gt_pred_mask(self, gt_masks, pred_masks):
		np_gt_masks = gt_masks.data.cpu().numpy()
		_, pred_masks = torch.max(pred_masks.data, 1)
		np_pred_masks = pred_masks.data.cpu().numpy()
		# np_pred_masks_ = np_pred_masks.copy()

		# print (np.max(np_gt_masks), np.max(np_pred_masks))
		np_gt_masks[np_gt_masks > 1] = 1
		np_pred_masks[np_pred_masks > 1] = 1
			
		for batch_id in range(np_gt_masks.shape[0]):
			self.update_single_gt_pred_mask(np_gt_masks[batch_id], 
											np_pred_masks[batch_id])

	def update_single_gt_pred_mask(self, np_gt_mask, np_pred_mask):
		for i in range(self.num_categories):
			for j in range(self.num_categories):
				gt_is_cate_i = (np_gt_mask == i)
				pred_is_cate_j = (np_pred_mask == j)
				self.confusion_matrix[i][j] += np.sum(np.logical_and(gt_is_cate_i, pred_is_cate_j).astype(int))

	def update_num_gt(self):
		for i in range(self.num_categories):
			self.num_gt[i] = np.sum(self.confusion_matrix[i, :])

	def update_pixel_accuracy(self):
		self.pixel_accuracy = np.matrix.trace(self.confusion_matrix) / np.sum(self.num_gt)

	def update_accuracy(self):
		for i in range(self.num_categories):
			self.accuracy[i] = self.confusion_matrix[i][i] / self.num_gt[i]

	def update_IoU(self):
		for i in range(self.num_categories):
			pred_on_i = sum([self.confusion_matrix[j][i] for j in range(self.num_categories)])
			self.IoU[i] = self.confusion_matrix[i][i] / (self.num_gt[i] + pred_on_i - self.confusion_matrix[i][i])

	def update_mean_accuracy(self):
		self.mean_accuracy = np.sum(self.accuracy) / self.num_categories

	def update_mean_IoU(self):
		self.mean_IoU = np.sum(self.IoU) / self.num_categories
		
	def summarize(self):
		self.update_num_gt()
		self.update_pixel_accuracy()
		self.update_accuracy()
		self.update_IoU()
		self.update_mean_accuracy()
		self.update_mean_IoU()
		

	def get_summarization(self):
		self.summarize()

		message = "Pixel acc = {:.4f}, mean acc = {:.4f}, mean IoU = {:.4f}"\
				.format(self.pixel_accuracy, self.mean_accuracy, self.mean_IoU)
		message += " ||| " + "{}: acc = {:.4f}, IoU = {:.4f}".format(self.cate_names[1], self.accuracy[1], self.IoU[1])
		message += " ||| " + "{}: acc = {:.4f}, IoU = {:.4f}".format(self.cate_names[0], self.accuracy[0], self.IoU[0])
		return message

class SegmentationMask():
	def __init__(self, mask_type, itensity_scale=255):		
		self.mask_type = mask_type
		self.segmentation_mask_path = None
		self.image_path_ids = {}
		self.itensity_scale = itensity_scale


	def save_batch_mask(self, image_paths, pred_mask):
		batch_size_ = pred_mask.size(0)
		_, pred_mask = torch.max(pred_mask.data, 1)
		np_pred_mask = pred_mask.data.cpu().numpy()

		# print (np.max(np_pred_mask), np.min(np_pred_mask))

		for batch_id in range(batch_size_):
			image_path = image_paths[batch_id].replace('/', '_').split('.')[0]
			self.update_image_path_ids(image_path)

			mask = Image.fromarray((np_pred_mask[batch_id] * self.itensity_scale).astype('uint8'), 'L')
			mask_path = self.get_mask_path(image_path)
			# print (mask_path)
			mask.save(mask_path)


	def set_segmentation_mask_path(self, ckpt_path, loader):
		model_name = ckpt_path.split(CKPT_PATH)[-1][:-5]
		self.segmentation_mask_path = alos.gen_dir("{}{}_on_{}_{}".format(RESULT_PATH, model_name, loader.name, self.mask_type))

	def update_image_path_ids(self, image_path):
		if image_path not in self.image_path_ids:
			self.image_path_ids[image_path] = 0
		else:
			self.image_path_ids[image_path] += 1

	def get_mask_path(self, image_path):
		return "{}{}_{}.png".format(
			self.segmentation_mask_path,
			image_path,
			self.image_path_ids[image_path])



