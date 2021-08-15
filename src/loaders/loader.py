from constants import *
from loaders.dataset import *

class Loader():
	def __init__(self, ann_file, batch_size, is_training, input_size=INPUT_SIZE):
		self.batch_size = batch_size
		self.name = ann_file.split('.')[0]
		self.is_training = is_training
		dataset = FoodDataset(ann_file=ann_file,
								transform=transforms.Compose([
									Rescale((input_size, input_size)),
									Rotate(is_training),
									ToNumpy(),
									AddGaussianNoise(is_training),
									ToTensor(),
									Normalize(IMAGENET_MEAN, IMAGENET_STD),
									]))

		self.loader = DataLoader(dataset, batch_size=self.batch_size,
										shuffle=is_training, num_workers=4)
		self.length = len(self.loader)
		print ("Number of batches: ", self.length)


class BatchData():
	def __init__(self, batch_data):
		self.image_path = batch_data[0]
		self.image = batch_data[1]
		self.counting = batch_data[2]
		if len(batch_data) > 3:
			self.half_polygon = batch_data[3] 
			self.full_polygon = batch_data[4]

			self.left_polygon = batch_data[5]
			self.bottom_left_polygon = batch_data[6]
			self.bottom_polygon = batch_data[7]
			self.bottom_right_polygon = batch_data[8]
			self.right_polygon = batch_data[9]
			self.top_right_polygon = batch_data[10]
			self.top_polygon = batch_data[11]
			self.top_left_polygon = batch_data[12]