from constants import *
from collections import defaultdict, OrderedDict



class Dataset(Dataset):
	def __init__(self, ann_file):
		self.ann_file = DATA_PATH + 'train_val/' + ann_file
		self.data_path = DATA_PATH + 'image/'
		self.dataset = list(open(self.ann_file, 'r'))
		self.data_length = len(self.dataset)
		self.image_path_ids = defaultdict(lambda : -1, OrderedDict())
		self.INPUT_SIZE = (VGG_INPUT_SIZE, VGG_INPUT_SIZE)

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		return self.extract_content(self.dataset[idx])

	def extract_content(self, content):
		[image_path, counting, aug_methods] = content.strip().split(',,,')
		aug_methods = json.loads(aug_methods)['aug_flip']

		content = OrderedDict()
		content['image_path'] = image_path
		self.image_path_ids[image_path] += 1
		content['result_image_path'] = image_path[:-4].replace('/', '_') + '_' + str(self.image_path_ids[image_path]) + '.png'
		content['org_image'] = self.get_image(image_path, aug_methods)
		content['processed_image'] = self.process_image(content['org_image'])
		content['counting'] = int(counting)

		return content

	def get_image(self, image_path, aug_methods):
		image = Image.open(self.data_path + image_path).convert("RGB")
		for aug_method in aug_methods:
			image = image.transpose(Image.__getattribute__(aug_method))
		return image

	def process_image(self, pil_image):
		image = np.array(pil_image.resize(self.INPUT_SIZE, Image.ANTIALIAS)) / 255
		image = image.transpose(2, 0, 1)
		image = torch.from_numpy(image).float()
		for t, m, s in zip(image, IMAGENET_MEAN, IMAGENET_STD):
			t.sub_(m).div_(s)
		image = torch.unsqueeze(image, 0)
		return image