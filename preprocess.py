from pydicom import dcmread
import pydicom
import gdcm
from PIL import Image
import PIL.ImageOps 
import datetime
import matplotlib.pyplot as plt
from natsort import natsorted
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import png
import shutil
import numpy as np
import os
from skimage import io, exposure, data
import cv2
from tqdm import tqdm
from skimage import img_as_float
import math
import pandas as pd
import configparser


# Create a ConfigParser object
config = configparser.ConfigParser()

# Read the external text file
config.read('config.txt')


dicom_dir = config.get('variables', 'dicom_dir')

png_dir = config.get('variables', 'png_dir')

png_mask_dir = config.get('variables', 'png_mask_dir')

processed_image_dir = config.get('variables', 'processed_image_dir')

mask_dir = config.get('variables', 'mask_dir')

processed_mask_dir = config.get('variables', 'processed_mask_dir')

num_cases = int(config.get('variables', 'num_cases'))


# move dicom files to parent dir
print('moving dicoms_to parent ...')
def move_to_parent(case):
    pth = []
    for root, dirs, files in os.walk(case, topdown=False):
        for name in files:
            pth.append(os.path.join(root, name))
        parent = case
    
    for i in pth:
        shutil.move(i, parent)



cases =[] 
for i in range(1,num_cases):
    
	cases.append(f'{dicom_dir}/case_{i}')

for x in cases:
    move_to_parent(x)





# save dicom files as png images
print('saving dicoms as png ...')

def resize_image(im, dim):
    dims = [im.size[0], im.size[1]]
    factor = dim/max(dims)
    resized_im = im.resize((round(im.size[0]*factor), round(im.size[1]*factor)))
 
    #Setting the points for cropped image
    left = -1*((dim-resized_im.size[0])/2)
    top = -1*((dim- resized_im.size[1])/2)
    right = (dim+resized_im.size[0])/2
    bottom = (dim+resized_im.size[1])/2
    resized =  resized_im.crop((left, top, right, bottom))
    # print(resized.shape)
    return resized

counter = list(range(1,num_cases))   
for i in tqdm(counter):
    path = f'{dicom_dir}/CASE_{i}/'
    ids  = natsorted(glob.glob(f'{path}/*.dcm'))
    
    for j in range(len(ids)):
        slice_id = ids[j]
        image_id = slice_id.split('/')[-1][:-4]
        dcm_path = slice_id
        dcm = dcmread(dcm_path).pixel_array.astype(float)
        dcm  = (np.maximum(dcm,0) / dcm.max()) * 255.0
        dcm = np.uint8(dcm)
        img = Image.fromarray(dcm)
        img = resize_image(img,256)

        directory = f'{png_dir}/CASE_{i}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        img.save(f"{directory}/{image_id}.png", format="png")


counter = list(range(1,num_cases))   
for i in tqdm(counter):
    path = f'{mask_dir}/CASE_{i}/'
    ids  = natsorted(glob.glob(f'{path}/*.dcm'))
    
    for j in range(len(ids)):
        slice_id = ids[j]
        image_id = slice_id.split('/')[-1][:-4]
        dcm_path = slice_id
        dcm = dcmread(dcm_path).pixel_array.astype(float)
        dcm  = (np.maximum(dcm,0) / dcm.max()) * 255.0
        dcm = np.uint8(dcm)
        img = Image.fromarray(dcm)
        img = resize_image(img,256)

        directory = f'{png_mask_dir}/CASE_{i}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        img.save(f"{directory}/{image_id}.png", format="png")




# preprocess images 
print('procssing images ...')

def intensity_transfer(image,mu,sigma):
    x_mean, x_std = cv2.meanStdDev(image)
    x_mean = np.hstack(np.around(x_mean,2))
    x_std = np.hstack(np.around(x_std,2))

    height, width= image.shape
    for i in range(0,height):
        for j in range(0,width):
            x = image[i,j]
            x = ((x-x_mean)*(sigma/x_std))+mu
            x = round(x.item())
            # boundary check
            x = 0 if x<0 else x
            x = 160 if x>160 else x
            image[i,j] = x 
    return image


counter = list(range(1,num_cases))   
for i in tqdm(counter):
    path = f'{png_dir}/CASE_{i}/'
    ids  = natsorted(glob.glob(f'{path}/*.png'))
    
    for j in range(len(ids)):
        slice_id = ids[j]

        image = cv2.imread(slice_id)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = intensity_transfer(image,mu = 46.19, sigma = 47.18)
        directory = f'{processed_image_dir}/CASE_{i}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        png.from_array(image, 'L').save(f"{directory}/{j}.png")


# process masks
print('procssing masks ...')
def process_mask(image):
    image = image.convert('L')
    image = image.point(lambda x: 0 if x<255 else 1, '1')
    return np.array(image)

counter = list(range(1,num_cases))   
for i in tqdm(counter):
    path = f'{png_mask_dir}/CASE_{i}/'
    ids  = natsorted(glob.glob(f'{path}/*.png'))
    
    for j in range(len(ids)):
        slice_id = ids[j]
        image_id = slice_id.split('/')[-1][:-4]
        image  = Image.open(slice_id)
        image = process_mask(image)
        image = Image.fromarray(image)
        image = resize_image(image, 256)
        directory = f'{processed_mask_dir}/CASE_{i}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        image.save(f"{directory}/{j}.png", format="png")






