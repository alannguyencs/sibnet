from constants import *
from loaders.loader import BatchData
from evals.counting import EvalCounting, DetailCounting
from evals.segmentation import EvalBinarySegmentation, SegmentationMask
from utils import network, util_log, util_os
from utils import sibnetcnt as post_processing

class Model():
	def __init__(self, arch):
		self.model = arch.to(device)
		self.name = arch.name

		self.opt_model_path = None
		self.last_model_path = None

		self.epoch_id = None
		self.num_epoch = 128
		self.max_lr = 8e-4
		self.base_lr = 1e-6
		self.lr_step = 16
		self.lr = self.max_lr	


		self.counting_criterion = nn.L1Loss().to(device)
		self.segmentation_criterion = network.cross_entropy_loss2d
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
		self.polygon_version = None

		self.loss = None
		self.seg_loss = None
		self.sib_loss = None
		self.cnt_loss = None

		self.eval_rc = None
		self.eval_seg = None
		self.eval_sib = None

		self.opt_eval_rc = 0
		self.epoch_loss_ = 0  #for each of train or val loss
		self.epoch_loss = 0   #sum of train and val loss
		self.min_epoch_loss = 1e5
		self.current_time = time.time()

		self.training_phase = 0
		self.log = None


	def train(self, train_loader, val_loader, model_id=0, polygon_version='polygon50'):
		self.set_opt_model_path_and_log(train_loader, model_id)
		self.set_log(self.opt_model_path)
		self.polygon_version = polygon_version

		for epoch_id in range(1, self.num_epoch + 1):
			self.epoch_id = epoch_id
			self.show_epoch_info()
			self.epoch_loss = 0

			self.run_epoch(train_loader)
			self.run_epoch(val_loader)
			
			self.save_checkpoint()			
			self.update_lr()

	def test(self, ckpt_path, test_loader):
		self.load_ckpt(ckpt_path)
		self.run_epoch(test_loader)
		self.eval_rc.show_detail_counting_result()

		self.save_masks(ckpt_path, test_loader)

	def test_counting(self, ckpt_path, test_loader):
		self.load_ckpt(ckpt_path)
		self.model.eval()
		self.eval_rc = EvalCounting()
		detail_rc = DetailCounting(ckpt_path, test_loader)

		for _, batch in enumerate(test_loader.loader):
			batch_data = BatchData(batch)
			image_paths = batch_data.image_path
			image = batch_data.image.to(device)
			gt_counting = batch_data.counting.to(device)

			if image.size(0) != test_loader.batch_size: break

			_, pred_counting, _ = self.model(image)
			self.eval_rc.update_batch_gt_pred_counting(gt_counting, pred_counting)
			detail_rc.write_batch_result(image_paths, pred_counting)

		print(self.eval_rc.get_summarization())
		# self.eval_rc.show_detail_counting_result()


	def run_epoch(self, loader):
		if loader.is_training: self.model.train()
		else: self.model.eval()
		self.refresh_eval_and_loss()

		for _, batch in tqdm(enumerate(loader.loader)):
			self.run_batch(loader, batch)
		
		self.epoch_loss_ /= loader.length
		self.epoch_loss += self.epoch_loss_
		self.summary_epoch(loader)


	def run_batch(self, loader, batch):
		batch_data = BatchData(batch)
		image = batch_data.image.to(device)
		gt_counting = batch_data.counting.to(device)
		gt_mask = batch_data.half_polygon.to(device)
		gt_masks = [batch_data.full_polygon, batch_data.left_polygon, batch_data.top_polygon, 
											batch_data.top_left_polygon, batch_data.top_right_polygon]

		for mask_id in range(5): gt_masks[mask_id] = gt_masks[mask_id].to(device)
		if image.size(0) != loader.batch_size: return

		pred_mask, pred_counting, pred_masks = self.model(image)
		self.eval_rc.update_batch_gt_pred_counting(gt_counting, pred_counting) #off counting
		self.eval_seg.update_batch_gt_pred_mask(gt_mask, pred_mask)
		for i in range(5): self.eval_sib[i].update_batch_gt_pred_mask(gt_masks[i], pred_masks[i])

		self.cnt_loss = self.counting_criterion(pred_counting, gt_counting) / 5.0  #off counting                                                   #off counting
		self.seg_loss = self.segmentation_criterion(pred_mask, gt_mask)
		
		#=== CROSS ENTROPY LOSS =========================================
		weight_unit = [1.0, 0.25, 0.25, 0.25, 0.25]
		weight_loss = [network.get_class_weight(gt_masks[i]).to(device) for i in range(5)]
		self.sib_loss = sum([weight_unit[i] * self.segmentation_criterion(pred_masks[i],\
														 gt_masks[i],\
														  weight_loss[i])
														  for i in range(5)])

		#==== DICE LOSS ===========================================
		alpha, beta = 0.001, 0.25
		#step 1: include above cross entropy loss: alpha
		#step 2: implement the following dice loss: beta

		self.sib_loss = self.segmentation_criterion(pred_masks[0], gt_masks[0])
		# print ('self.sib_loss', self.sib_loss.size())
		(B, H, W) = gt_masks[0].size()
		softmax = nn.Softmax(dim=3)
		for sib_id in range(1, 5):
			weight_loss = network.get_class_weight(gt_masks[sib_id]).to(device)
			self.sib_loss += alpha * self.segmentation_criterion(pred_masks[sib_id], gt_masks[sib_id], weight_loss)

			gt_onehot = torch.zeros(B * H * W, 2).to(device) #one hot mask
			gt_labels = gt_masks[sib_id].view(-1, 1)
			gt_onehot.scatter_(1, gt_labels, 1) #B * H * W, 2

			pred_sib = pred_masks[sib_id].transpose(1,2).transpose(2,3) #BCHW -> BHWC
			pred_prob = softmax(pred_sib)
			
			gt_onehot =  gt_onehot.contiguous()
			pred_prob = pred_prob.contiguous()

			gt_onehot_T = gt_onehot.contiguous().view(1, -1) #1 * M
			gt_onehot_ =  gt_onehot.contiguous().view(-1, 1) #M * 1
			pred_prob_T = pred_prob.contiguous().view(1, -1) #1 * M
			pred_prob_ =  pred_prob.contiguous().view(-1, 1) #M * 1

			dice_loss_ = (torch.matmul(gt_onehot_T, gt_onehot_) + torch.matmul(pred_prob_T, pred_prob_))\
			                 / (2. * torch.matmul(gt_onehot_T, pred_prob_))
			self.sib_loss += beta * dice_loss_.squeeze()

		#dice loss for half segmentation
		gt_onehot = torch.zeros(B * H * W, 2).to(device) #one hot mask
		gt_labels = gt_mask.view(-1, 1)
		gt_onehot.scatter_(1, gt_labels, 1) #B * H * W, 2

		pred_mask_ = pred_mask.transpose(1,2).transpose(2,3) #BCHW -> BHWC
		pred_prob = softmax(pred_mask_)
		
		gt_onehot =  gt_onehot.contiguous()
		pred_prob = pred_prob.contiguous()

		gt_onehot_T = gt_onehot.contiguous().view(1, -1) #1 * M
		gt_onehot_ =  gt_onehot.contiguous().view(-1, 1) #M * 1
		pred_prob_T = pred_prob.contiguous().view(1, -1) #1 * M
		pred_prob_ =  pred_prob.contiguous().view(-1, 1) #M * 1

		dice_loss_ = (torch.matmul(gt_onehot_T, gt_onehot_) + torch.matmul(pred_prob_T, pred_prob_))\
		                 / (2. * torch.matmul(gt_onehot_T, pred_prob_))

										
		self.pick_loss()
		self.backprop(loader, self.loss)


	def show_epoch_info(self):
		self.log.info ('\nEpoch [{}/{}], lr {:.6f}, runtime {:.3f}'.format(self.epoch_id, self.num_epoch, self.lr, time.time()-self.current_time))
		self.current_time = time.time()

	def summary_epoch(self, loader):
		self.log.info("{}: loss={:.6f}".format(loader.name, self.epoch_loss))
		self.log.info(self.eval_rc.get_summarization())
		self.log.info(self.eval_seg.get_summarization())
		for i in range(5): self.log.info(self.eval_sib[i].get_summarization())

	def save_checkpoint(self):
		if self.min_epoch_loss > self.epoch_loss:
			self.min_epoch_loss = self.epoch_loss
			torch.save(self.model.state_dict(), self.opt_model_path)
			self.log.info ("checkpoint saved")

		if (self.epoch_id + 1) % self.lr_step == 0:
			torch.save(self.model.state_dict(), self.last_model_path)


	def refresh_eval_and_loss(self):
		self.eval_rc = EvalCounting()
		self.eval_seg = EvalBinarySegmentation()
		self.eval_sib = [EvalBinarySegmentation() for _ in range(5)]
		self.epoch_loss_ = 0

	def pick_loss(self):
		self.loss = self.cnt_loss + self.seg_loss + self.sib_loss
		self.epoch_loss_ += self.loss.item()


	def backprop(self, loader, loss):
		if loader.is_training:
			self.optimizer.zero_grad()
			self.loss.backward()
			self.optimizer.step()

	def set_opt_model_path_and_log(self, train_loader, model_id):
		model_name = self.name + '_' + train_loader.name + '_' + str(model_id)
		self.opt_model_path = CKPT_PATH + model_name + '.ckpt'
		self.last_model_path = CKPT_PATH + model_name + '_last.ckpt'

		
	def set_log(self, ckpt_path):
		model_name = ckpt_path.split(CKPT_PATH)[-1].split('.ckpt')[0]
		log_path = LOG_PATH + model_name + '.txt'
		self.log = util_log.Allog(log_path)
	
	
	def load_ckpt(self, ckpt_path="opt"):
		if ckpt_path == "opt": ckpt_path = self.opt_model_path
		if ckpt_path == "last": ckpt_path = self.last_model_path
		self.model.load_state_dict(torch.load(ckpt_path))

	def update_lr(self):
		if self.epoch_id % self.lr_step == 0: self.max_lr *= 0.88
		self.lr = self.max_lr - (self.max_lr - self.base_lr) * (self.epoch_id % self.lr_step) / self.lr_step
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = self.lr

	def save_masks(self, ckpt_path, loader):
		self.model.eval()
		segmentation_mask = SegmentationMask('half')
		segmentation_mask.set_segmentation_mask_path(ckpt_path, loader)

		mask_types = ['full', 'left', 'top', 'top_left', 'top_right']
		segmentation_masks = [SegmentationMask(mask_types[i]) for i in range(5)]
		for i in range(5): segmentation_masks[i].set_segmentation_mask_path(ckpt_path, loader)
		
		for _, batch in enumerate(loader.loader):
			batch_data = BatchData(batch)
			image = batch_data.image.to(device)
			image_path = batch_data.image_path

			pred_mask, pred_counting, pred_masks = self.model(image)

			segmentation_mask.save_batch_mask(image_path, pred_mask)
			for i in range(5): 
				segmentation_masks[i].save_batch_mask(image_path, pred_masks[i])


	def save_sibling_detection_map(self, ckpt_path, loader):
		self.load_ckpt(ckpt_path)
		self.model.eval()

		model_name = ckpt_path.split(CKPT_PATH)[-1][:-5]
		mask_dir = util_os.gen_dir("{}{}_on_{}_sibling".format(RESULT_PATH, model_name, loader.name), True)
		image_path_ids = defaultdict(lambda : -1, OrderedDict())
				
		for _, batch in tqdm(enumerate(loader.loader)):
			batch_data = BatchData(batch)
			image = batch_data.image.to(device)
			image_paths = batch_data.image_path
			if image.size(0) != loader.batch_size: return

			_, _, pred_masks = self.model(image)
			pred_np_masks = [pred_masks[i].data.cpu().numpy() for i in range(5)]

			for batch_id in range(loader.batch_size):
				image_path = image_paths[batch_id].replace('/', '_').split('.')[0]
				image_path_ids[image_path] += 1

				for mask_id in range(5):
					sibling_mask = pred_np_masks[mask_id][batch_id][1]
					plt.imshow(sibling_mask, interpolation='none') #foreground probability
					plt.savefig("{}{}_{}_{}.png".format(mask_dir, image_path, image_path_ids[image_path], mask_id))

	def save_instance_map(self, ckpt_path, loader):
		self.load_ckpt(ckpt_path)
		self.model.eval()

		model_name = ckpt_path.split(CKPT_PATH)[-1][:-5]
		result_path = "{}{}_on_{}".format(RESULT_PATH, model_name, loader.name)
		print ("The results is found at {}".format(result_path))
		mask_dir = util_os.gen_dir(result_path, True)
		image_path_ids = defaultdict(lambda : -1, OrderedDict())
				
		for _, batch_data in tqdm(enumerate(loader.loader)):
			image_paths = batch_data[0]
			image = batch_data[1].to(device)			
			if image.size(0) != loader.batch_size: return

			pred_half, pred_counting, pred_masks = self.model(image)
			_, pred_half = torch.max(pred_half.data, 1)
			np_pred_half = pred_half.data.cpu().numpy()
			pred_counting = np.round(pred_counting.data.cpu().numpy(), 0)
			pred_np_masks = [None for _ in range(5)]
			

			#full segmentation
			_, pred_mask = torch.max(pred_masks[0].data, 1)
			pred_np_masks[0] = pred_mask.data.cpu().numpy()

			#sibling
			for mask_id in range(1, 5):
				softmax = nn.Softmax(dim=3)
				pred_sib = pred_masks[mask_id].transpose(1,2).transpose(2,3) #BCHW -> BHWC
				pred_prob = softmax(pred_sib)
				pred_mask_ = pred_prob.data.cpu().numpy()[:, :, :, 1]
				pred_mask_[pred_mask_ >= 0.94] = 1
				pred_mask_[pred_mask_ < 0.94] = 0
				pred_np_masks[mask_id] = pred_mask_

			for batch_id in range(loader.batch_size):
				image_path = image_paths[batch_id].replace('/', '_').split('.')[0]
				image_path_ids[image_path] += 1
				mask_path = "{}{}_{}.png".format(mask_dir, image_path, image_path_ids[image_path])

				pred_counting_ = int(pred_counting[batch_id])
				pred_np_masks_ = [pred_np_masks[mask_id][batch_id] for mask_id in range(5)]

				pred_full = pred_np_masks_[0]
				pred_half = np_pred_half[batch_id]
				pred_half[pred_full==0] = 0
				pred_np_masks_[0] = pred_full + pred_half
				post_processing.save_instance_map(pred_np_masks_, mask_path, pred_counting_)


	def test_speed_instance_segmentation(self, ckpt_path, ann_file):
		from utils import image_preprocessing
		self.load_ckpt(ckpt_path)
		self.model.eval()
		ann_file = DATA_PATH + 'train_val/' + ann_file
		dataset = list(open(ann_file, 'r'))
		image_paths = []
		for line in dataset:
			[image_path, _, _] = line.strip().split(',,,')
			image_paths.append(DATA_PATH + 'image/' + image_path)

		time_loading_image, time_preprocessing, time_model, time_post = 0, 0, 0, 0
		for image_id, image_path in tqdm(enumerate(image_paths)):
			mask_path = "{}{}.png".format(BUFFER_PATH, image_id)
			t_0 = time.time()

			image = Image.open(image_path).convert('RGB')
			t_1 = time.time()

			image = image.resize((VGG_INPUT_SIZE, VGG_INPUT_SIZE), Image.ANTIALIAS)
			image = np.array(image) / 255.0
			image = image.transpose(2, 0, 1)
			image = torch.from_numpy(image).float()
			for t, m, s in zip(image, IMAGENET_MEAN, IMAGENET_STD):
				t.sub_(m).div_(s)
			image = torch.unsqueeze(image, 0).to(device)
			t_2 = time.time()

			pred_half, pred_counting, pred_masks = self.model(image)
			_, pred_half = torch.max(pred_half.data, 1)
			np_pred_half = pred_half.data.cpu().numpy()
			pred_counting = np.round(pred_counting.data.cpu().numpy(), 0)
			pred_np_masks = [None for _ in range(5)]
			for mask_id in range(5):
				_, pred_mask = torch.max(pred_masks[mask_id].data, 1)
				pred_np_masks[mask_id] = pred_mask.data.cpu().numpy()
			t_3 = time.time()

			pred_counting_ = int(pred_counting[0])
			pred_np_masks_ = [pred_np_masks[mask_id][0] for mask_id in range(5)]

			pred_full = pred_np_masks_[0]
			pred_half = np_pred_half[0]
			pred_half[pred_full==0] = 0
			pred_np_masks_[0] = pred_full + pred_half

			time_post += post_processing.save_instance_map(pred_np_masks_, mask_path, pred_counting_)
			
			time_loading_image += t_1 - t_0
			time_preprocessing += t_2 - t_1
			time_model += t_3 - t_2

		time_loading_image *= 1000 / len(image_paths)
		time_preprocessing *= 1000 / len(image_paths)
		time_model *= 1000 / len(image_paths)
		time_post *= 1000 / len(image_paths) 

		total_time = time_loading_image + time_preprocessing + time_model + time_post
		print ("FPS: {:.2f}".format(1000. / total_time))
		print ("FPS: {:.2f}".format(1000. / total_time))
		print ('Runtime: loading image | preprocessing | model | postprocessing | =  \
			{:.0f} ms-{:.0f}% | {:.0f} ms-{:.0f}% | {:.0f} ms-{:.0f}% | {:.0f} ms-{:.0f}%'
			.format(time_loading_image, 100*time_loading_image/total_time, 
				time_preprocessing, 100*time_preprocessing/total_time, 
				time_model, 100*time_model/total_time, time_post, 100*time_post/total_time))



	def test_instance_map_new_data(self, ckpt_path, image_dir):
		self.load_ckpt(ckpt_path)
		self.model.eval()

		model_name = ckpt_path.split(CKPT_PATH)[-1][:-5]
		result_path = "{}{}_on_{}".format(RESULT_PATH, model_name, 'new_samples')
		print ("The results is found at {}".format(result_path))
		mask_dir = util_os.gen_dir(result_path, True)

		data = glob.glob("{}/*".format(image_dir))
		for image_path in tqdm(data):
			image_name = image_path.split('/')[-1].split('.')[0]

			image = Image.open(image_path).convert('RGB').resize((256, 256), Image.ANTIALIAS)
			image = np.array(image) / 255.0
			image = image.transpose(2, 0, 1)
			image = torch.from_numpy(image).float()
			for t, m, s in zip(image, IMAGENET_MEAN, IMAGENET_STD):
				t.sub_(m).div_(s)
			image = torch.unsqueeze(image, 0).to(device)

			pred_half, pred_counting, pred_masks = self.model(image)
			_, pred_half = torch.max(pred_half.data, 1)
			np_pred_half = pred_half.data.cpu().numpy()
			pred_counting = np.round(pred_counting.data.cpu().numpy(), 0)
			pred_np_masks = [None for _ in range(5)]
			for mask_id in range(5):
				_, pred_mask = torch.max(pred_masks[mask_id].data, 1)
				pred_np_masks[mask_id] = pred_mask.data.cpu().numpy()

			pred_counting_ = int(pred_counting[0])
			pred_np_masks_ = [pred_np_masks[mask_id][0] for mask_id in range(5)]

			pred_full = pred_np_masks_[0]
			pred_half = np_pred_half[0]
			pred_half[pred_full==0] = 0
			pred_np_masks_[0] = pred_full + pred_half

			mask_path = "{}{}.png".format(mask_dir, image_name)
			post_processing.save_instance_map(pred_np_masks_, mask_path, pred_counting_)
	











