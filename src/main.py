import sys
# import runs
# from runs import *
import importlib

if __name__ == '__main__':
	arg = sys.argv
	model = importlib.import_module('runs.{}'.format(arg[1]))
	model.__getattribute__(arg[2])(arch_id=arg[3], model_id=arg[4], food_type=arg[5])