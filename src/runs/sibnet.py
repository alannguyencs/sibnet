from constants import *
from models import sibnet
from loaders import loader
from archs import sibnet_6 as arch




def train(arch_id, model_id, food_type):
	val_loader = loader.Loader(ann_file='{}_val_100.txt'.format(food_type),
											batch_size=8,
											is_training=False)


	train_loader = loader.Loader(ann_file='{}_train_100.txt'.format(food_type),
												batch_size=16,
												is_training=True)
	

	model = sibnet.Model(arch.Arch())
	model.train(train_loader, val_loader, int(model_id))



def test(arch_id, model_id, food_type):
	ckpt_path = CKPT_PATH + "SIBNET_v{}_{}_train_100_{}.ckpt".format(arch_id, food_type, model_id)
	val_loader = loader.Loader(ann_file='{}_val_100.txt'.format(food_type),
											batch_size=3,
											is_training=False,
											input_size=256)

	model = sibnet.Model(arch.Arch())
	model.load_ckpt(ckpt_path)
	model.set_log(ckpt_path)
	model.test(ckpt_path, val_loader)

def test_counting(arch_id, model_id, food_type):
	ckpt_path = CKPT_PATH + "SIBNET_v{}_{}_train_100_{}.ckpt".format(arch_id, food_type, model_id)
	test_loader = loader.Loader(ann_file='{}_test_100.txt'.format(food_type),
											batch_size=3,
											is_training=False,
											input_size=256)
											

	model = sibnet.Model(arch.Arch())
	model.load_ckpt(ckpt_path)
	model.set_log(ckpt_path)
	model.test_counting(ckpt_path, test_loader)


def save_instance_map(arch_id, model_id, food_type):
	train_food_type = 'cookie'
	test_food_type = 'cookie'
	ckpt_path = CKPT_PATH + "SIBNET_v{}_{}_train_100_{}.ckpt".format(arch_id, train_food_type, model_id)
	test_loader = loader.Loader(ann_file='{}_test_100.txt'.format(test_food_type),
											batch_size=3,
											is_training=False,
											input_size=256)

	model = sibnet.Model(arch.Arch())
	model.load_ckpt(ckpt_path)
	model.set_log(ckpt_path)
	model.save_instance_map(ckpt_path, test_loader)


def test_speed(arch_id, model_id, food_type):
	ann_file='{}_test_100.txt'.format(food_type)
	ckpt_path = CKPT_PATH + "SIBNET_v{}_{}_train_100_{}.ckpt".format(arch_id, food_type, model_id)
	model = sibnet.Model(arch.Arch())
	model.test_speed_instance_segmentation(ckpt_path, ann_file)


def test_new_images(arch_id, model_id, food_type, image_dir):
	ckpt_path = CKPT_PATH + "SIBNET_v{}_{}_train_100_{}.ckpt".format(arch_id, food_type, model_id)
	model = sibnet.Model(arch.Arch())
	model.test_instance_map_new_data(ckpt_path, image_dir)
