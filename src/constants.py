import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import torch.nn.functional as F
from PIL import Image
from shutil import rmtree
import math
import queue
from shutil import copyfile
import sys
import platform
from random import shuffle
from tqdm import trange, tqdm
import colorsys
from pathlib import Path
from collections import defaultdict, OrderedDict
import json
import glob

project_path = "/home/tnguyenhu2/alan_project/"
DATA_PATH = project_path + '/data/foodcounting/'


SIBNET_PATH = project_path + "sibnet/"
CKPT_PATH = SIBNET_PATH + 'ckpt/'
RESULT_PATH = SIBNET_PATH + 'result/'
LOG_PATH = SIBNET_PATH + 'log/'


INPUT_SIZE = 256 
SEG_OUTPUT_SIZE = 256

IMAGENET_MEAN=[0.485, 0.456, 0.406]
IMAGENET_STD=[0.229, 0.224, 0.225]




COUNTING_DICT = {'cookie': 9, 'dimsum': 6, 'sushi': 6}
MASK_TYPES = ['half50', 'half75', 'full', 'left', 'top', 'top_left', 'top_right', 
		'half_polygon', 'full_polygon', 'left_polygon', 'bottom_left_polygon', 'bottom_polygon', 'bottom_right_polygon', 
		'right_polygon', 'top_right_polygon', 'top_polygon', 'top_left_polygon', 'dual_polygon',
		'polygon10', 'polygon20', 'polygon30', 'polygon40',
		'polygon50', 'polygon60', 'polygon70', 'polygon80', 'polygon90']   #PNG


#CUDA ENVIRONMENT
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

