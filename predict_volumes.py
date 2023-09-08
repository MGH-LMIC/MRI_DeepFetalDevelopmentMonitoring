import torch
import torchvision
# from dataset import MRIDataset
from torch.utils.data import DataLoader
import os
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from glob import glob
from pydicom import dcmread
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from natsort import natsorted
from PIL import Image
from dataset import MRIDataset
import configparser

# Create a ConfigParser object
config = configparser.ConfigParser()

# Read the external text file
config.read('config.txt')


processed_image_dir = config.get('variables', 'processed_image_dir')


processed_mask_dir = config.get('variables', 'processed_mask_dir')

num_cases = int(config.get('variables', 'num_cases'))


def upscale_image(im, dims, dim = 256):
    im = Image.fromarray(im)
    factor = (max(dims))/dim
    # print(factor*im.size[0], factor*im.size[1])
    resized_im = im.resize((round(im.size[0]*factor), round(im.size[1]*factor)))
    return np.array(resized_im)


def read_folder(image_dir):

	images = []
	for root, dirs, files in os.walk(image_dir, topdown=False):
	    for name in files:
	        images.append(os.path.join(root, name))

	return natsorted(images)


def get_case(idx):
    return [df.index[df['slice'] == i].item() for i in df['slice'] if f'CASE_{idx}/' in i]


val_transform = A.Compose(
    [
        # A.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH),
        A.Normalize(
              mean=[0.0,0.0,0.0],
              std=[1.0,1.0,1.0],
              max_pixel_value=255.0),
        ToTensorV2(),
            
    ]
    
    )

DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'

model_path = './model_256_split_0.pt'
# model_path = f'./{folder}/model_160_split_0.pt'
model = torch.load(model_path).to(DEVICE)

batch_size = 4
num_workers = 2
pin_memory = True


# each case tested separately

# first need to calculate the ground truth volumes of all cases


df = pd.read_csv('./all_new_afv_volumes_small.csv')

# convert the string representation of tuples back to tuples
df["new_sizes"] = df["new_sizes2"].apply(lambda x: tuple(eval(x)))

# iterate over cases
print('calculating GT volumes ...')
case_vol = {}
for i in tqdm(range(1,num_cases)):
	gt_masks_path = f'{processed_mask_dir}/CASE_{i}/'


	# read and upscale masks 
	masks = read_folder(gt_masks_path)
	slices_vol = []
	for mask in masks:
	    msk = np.array(Image.open(mask))
	    idx = get_case(i)
	    idx = idx[1]
	    dims = df["new_sizes"][idx]
	    msk = upscale_image(msk, dims, dim = 256)

	    # calculate white area
	    area = np.sum(msk == 255)

	    # get voxel size , calculate volume
	    voxel = df['voxels'][i]
	    vol = (voxel* area)/1000

	    # save gt volumes in a list
	    slices_vol.append(vol)

	# sum up slice volumes
	case_vol[f'CASE_{i}'] = sum(slices_vol) #0.12 + (sum(slices_vol))*1.031
    
print('calculating pred volumes ...')
pred_case_vol = {}    
for i in tqdm(range(1,num_cases)):
	image_path = f'{processed_image_dir}/CASE_{i}'

	pred_slices_vol = []

	images = read_folder(image_path)
	for image in images:
	    idx = get_case(i)
	    idx = idx[1]
	    img = np.array(Image.open(image).convert('RGB'))
	    augmentations = val_transform(image = img)
	    image = augmentations['image']
	    x = image.to(DEVICE)

	    # read slices and predict masks
	    with torch.no_grad():

	        preds = torch.sigmoid(model(x.unsqueeze(0)))
	        preds = (preds > 0.5).float().squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)

	        preds = upscale_image(preds, dims, dim = 256) 

	        # print(preds.shape)

	        pred_area = np.sum(preds==1)
	        # get voxel size , calculate volume
	        voxel = df['voxels'][idx]
	        vol = (voxel* pred_area)/1000

	        # save gt volumes in a list
	        pred_slices_vol.append(vol)

	pred_case_vol[f'CASE_{i}'] = sum(pred_slices_vol) #0.12 + (sum(pred_slices_vol))*1.031




# finally combine the gt and predicted volumes in a csv file
new_df = pd.DataFrame(case_vol.items(), columns = ['Case','Ground_truth_vol'])
new_df['Predicted_vol'] = pred_case_vol.values()

new_df.to_csv('fetal_vol_GT_AI.csv', index = False)


