from constants import *
from models import sibnet
from loaders import loader
from archs import sibnet_6 as arch
import sys
import config



def train():
	val_loader = loader.Loader(ann_file='{}_val_100.txt'.format(config.food_type),
											batch_size=8,
											is_training=False)


	train_loader = loader.Loader(ann_file='{}_train_100.txt'.format(config.food_type),
												batch_size=16,
												is_training=True)
	

	model = sibnet.Model(arch.Arch())
	model.train(train_loader, val_loader)

def test():
	test_loader = loader.Loader(ann_file='{}_test_100.txt'.format(config.food_type),
											batch_size=3,
											is_training=False,
											input_size=256)

	model = sibnet.Model(arch.Arch())
	model.load_ckpt(config.ckpt_path)
	model.set_log(config.ckpt_path)
	model.save_instance_map(config.ckpt_path, test_loader)



if __name__ == '__main__':
	arg = sys.argv[1]
	if arg == 'train': train()
	if arg == 'test': test()

