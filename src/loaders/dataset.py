from constants import *

SIBLING_MASKS = ['half_polygon', 'full_polygon', 'left_polygon', 'bottom_left_polygon', 'bottom_polygon', 'bottom_right_polygon',
'right_polygon', 'top_right_polygon', 'top_polygon', 'top_left_polygon']


class FoodDataset(Dataset):
	def __init__(self, ann_file, transform=None):
		self.ann_file = DATA_PATH + 'train_val/' + ann_file
		self.data_path = DATA_PATH + 'images/'
		self.mask_path = DATA_PATH + 'mask/'
		self.transform = transform
		self.dataset = list(open(self.ann_file, 'r'))

		print ("data_length:", len(self.dataset))

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		sample = self.extract_content(self.dataset[idx])
		if self.transform:
			sample = self.transform(sample)
		
		sample_data = tuple([s_data for s_data in sample.values()])
		return sample_data


	def extract_content(self, content_):
		[image_path, counting, aug_methods] = content_.strip().split(',,,')[:3]
		aug_methods = json.loads(aug_methods)['aug_flip']

		content = OrderedDict()
		content['image_path'] = image_path
		content['image'] = self.get_image(image_path, aug_methods)
		content['counting'] = int(counting)

		for mask_type in SIBLING_MASKS:
			mask_path = self.mask_path + mask_type + '/' + image_path.replace('.jpg', '.png')
			content[mask_type] = Image.open(mask_path) if os.path.isfile(mask_path) else Image.new('L', (256, 256))

		for aug_method in aug_methods:
			for mask_type in SIBLING_MASKS:
				content[mask_type] = content[mask_type].transpose(Image.__getattribute__(aug_method))
			if 'LEFT_RIGHT' in aug_method: self.switch_left_right(content)
			if 'TOP_BOTTOM' in aug_method: self.switch_top_bottom(content)

		return content


	def get_image(self, image_path, aug_methods):
		image = Image.open(self.data_path + image_path).convert("RGB")
		for aug_method in aug_methods:
			image = image.transpose(Image.__getattribute__(aug_method))
		return image

	
	def switch_left_right(self, content):
		def swap(keyword_1, keyword_2):
			x = content[keyword_1]
			content[keyword_1] = content[keyword_2]
			content[keyword_2] = x
		
		swap('left_polygon', 'right_polygon')
		swap('bottom_left_polygon', 'bottom_right_polygon')
		swap('top_left_polygon', 'top_right_polygon')

	def switch_top_bottom(self, content):
		def swap(keyword_1, keyword_2):
			x = content[keyword_1]
			content[keyword_1] = content[keyword_2]
			content[keyword_2] = x
		
		swap('top_polygon', 'bottom_polygon')
		swap('bottom_left_polygon', 'top_left_polygon')
		swap('bottom_right_polygon', 'top_right_polygon')



class Rescale(object):
	def __init__(self, output_size):
		assert isinstance(output_size, tuple)
		self.output_size = output_size

	def __call__(self, sample):
		image = sample['image']
		assert isinstance(image, Image.Image)
		sample['image'] = image.resize(self.output_size, Image.ANTIALIAS)
		return sample

class Rotate(object):
	def __init__(self, is_training):
		self.is_training = is_training

	def __call__(self, sample):
		if self.is_training:
			rotate_times = [0, 1, 2, 3]
			shuffle(rotate_times)
			for _ in range(rotate_times[0]):
				sample['image'] = sample['image'].transpose(Image.__getattribute__('ROTATE_90'))
				for mask_type in SIBLING_MASKS:
					sample[mask_type] = sample[mask_type].transpose(Image.__getattribute__('ROTATE_90'))

				sample_rotate_ = sample['left_polygon']
				sample['left_polygon'] = sample['top_polygon']
				sample['top_polygon'] = sample['right_polygon']
				sample['right_polygon'] = sample['bottom_polygon']
				sample['bottom_polygon'] = sample_rotate_

				sample_rotate_ = sample['top_left_polygon']
				sample['top_left_polygon'] = sample['top_right_polygon']
				sample['top_right_polygon'] = sample['bottom_right_polygon']
				sample['bottom_right_polygon'] = sample['bottom_left_polygon']
				sample['bottom_left_polygon'] = sample_rotate_
				

		return sample


class ToNumpy(object):

	def __call__(self, sample):
		sample['image'] = np.array(sample['image'])
		sample['counting'] = np.array([sample['counting']], dtype='int64')
		for sample_key in sample.keys():
			if sample_key in MASK_TYPES:
				sample[sample_key] = np.array(sample[sample_key])


		return sample

class AddGaussianNoise(object):
	def __init__(self, is_training):
		self.is_training = is_training
		self.gaussian_mean = 0.0
		self.gaussian_std = 5.0

	def __call__(self, sample):
		if self.is_training:
			image = sample['image']
			gaussian = np.random.normal(self.gaussian_mean, self.gaussian_std, image.shape).astype(np.float64)
			sample['image'] = image + gaussian

		return sample

class ToTensor(object):
	def __init__(self, classification_counting=False):
		self.classification_counting = classification_counting

	def __call__(self, sample):
		image, counting = sample['image'], sample['counting']
		image = image.transpose((2, 0, 1))
		sample['image'] = torch.from_numpy(image).float()
		sample['counting'] = torch.from_numpy(counting).long() - 1 if self.classification_counting \
							else torch.from_numpy(counting).float()
		
		for sample_key in sample.keys():
			if sample_key in MASK_TYPES:
				normalized_mask = torch.from_numpy(sample[sample_key] / 255.0)
				sample[sample_key] = normalized_mask.long()

		return sample


class Normalize(object):
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def __call__(self, sample):
		image = sample['image']
		image = image / 255.0
		for t, m, s in zip(image, self.mean, self.std):
			t.sub_(m).div_(s)

		sample['image'] = image
		return sample






