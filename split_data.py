from PIL import Image
import PIL.ImageOps 
import datetime
from natsort import natsorted
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import shutil
import numpy as np
import os
from tqdm import tqdm
from skimage import img_as_float
import math
import pandas as pd
import random
from tqdm import tqdm
import os
import shutil
import configparser

# Create a ConfigParser object
config = configparser.ConfigParser()

# Read the external text file
config.read('config.txt')


def read_folder(image_dir,  seed = 123):

	random.seed(seed)

	images = []
	for root, dirs, files in os.walk(image_dir, topdown=False):
	    for name in files:
	        images.append(os.path.join(root, name))

	return natsorted(images)

def get_splits(total):
	    train = int(total*.8)
	    rest = total - train
	    test = train + (rest//2)
	    
	    return train, test

image_dir = config.get('variables', 'processed_image_dir')

all_images = read_folder(image_dir)

mask_dir = config.get('variables', 'processed_mask_dir')

all_masks = read_folder(mask_dir)


image_dist = f'./mixed_images'


if not os.path.exists(image_dist):
    os.makedirs(image_dist)
    
for i in range(len(all_images)):
    shutil.copy(all_images[i], os.path.join(image_dist,f'{i}.png'))


mask_dist = f'./mixed_masks'

if not os.path.exists(mask_dist):
    os.makedirs(mask_dist)
    
for i in range(len(all_masks)):
    shutil.copy(all_masks[i], os.path.join(mask_dist,f'{i}.png'))



dist = f'./split_0'
# print(dist)
if not os.path.exists(dist):
    os.makedirs(dist)


for idx in range(2):

	folder = ['images', 'masks']

	image_path = f'./mixed_{folder[idx]}'

	cases = []
	for i in natsorted(os.listdir(image_path)):
	    cases.append(i)

	    
	print(len(cases))

	splits = {}

	tr, ts= get_splits(len(cases))
	random.seed(99) # seed  = 25 
	for i in range(1):
	    random.shuffle(cases)
	    train = cases[:tr]    
	    test = cases[tr:ts]
	    val = cases[ts:]
	    splits[i] = (train, test, val)
	    print(len(train), len(val), len(test))


	    
	# folders = [f'train_{tag[idx]}', f'test_{tag[idx]}', f'val_{tag[idx]}' ]
	folders = [f'train_{folder[idx]}', f'test_{folder[idx]}', f'val_{folder[idx]}' ]
	for j in range(3):
	    folder2 = os.path.join(dist,folders[j])
	    
	    # print(folder)
	    if not os.path.exists(folder2):
	        os.makedirs(folder2)
	    for x in splits[i][j]:
	        source_dir = os.path.join(image_path,x)
	        # print(source_dir)
	        shutil.copy(source_dir,os.path.join(folder2,x))