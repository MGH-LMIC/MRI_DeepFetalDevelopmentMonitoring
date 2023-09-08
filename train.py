import os
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
# from model import UNET
# from focal_dice import FocalDiceLoss
# from model3 import Nested_UNet
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs)

import configparser

# Create a ConfigParser object
config = configparser.ConfigParser()

# Read the external text file
config.read('config.txt')

folder = config.get('variables', 'folder')

DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
def train_fn(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader)
    for batch_idx, (data,targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    return model
        
        


def main(resize, SPLIT):
    LEARNING_RATE = 1e-4
    CV = 19
    BATCH_SIZE = 4
    NUM_EPOCHS  = 100
    NUM_WORKERS = 2
    PIN_MEMORY  = True
    LOAD_MODEL = False
    TRAIN_IMG_DIR = f'./{folder}/split_0/train_images/'
    TRAIN_MASK_DIR = f'./{folder}/split_0/train_masks/'
    VAL_IMG_DIR = f'./{folder}/split_0/val_images/'
    VAL_MASK_DIR = f'./{folder}/split_0/val_masks/'
    IMAGE_HEIGHT  = resize
    IMAGE_WIDTH = resize
    train_transform = A.Compose(
        [
          # A.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH),
          A.Rotate(limit = 50, p = 1.0),
          A.HorizontalFlip(p=0.5),
          A.VerticalFlip(p=0.5),
          A.Normalize(
              mean=[0.0,0.0,0.0],
              std=[1.0,1.0,1.0],
              max_pixel_value=255.0),
            ToTensorV2(),
            
            
        ]
    
    )
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
    
    # model = UNET(in_channels=3, out_channels = 1).to(DEVICE)
    model = smp.Unet('vgg16', encoder_weights='imagenet').to(DEVICE)
    # model = Nested_UNet().to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss() #DiceLoss() #tgm.losses.DiceLoss()#nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY
    )
    
    filename = f'./{folder}/model_{resize}_split_{SPLIT}.pt'
    if LOAD_MODEL:
        load_checkpoint(torch.load(filename), model)
    
    scaler = torch.cuda.amp.GradScaler()
    
    dice_zero = 0
    for epoch in range(NUM_EPOCHS):
        print(f'Epoch: {epoch+1}, size: {resize}, split: {SPLIT}')
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        
        # check accuracy
        dice, acc = check_accuracy(val_loader, model, device = DEVICE)
        # save model
        if dice > dice_zero:
            print('=> Saving checkpoint ---------------------------------------------------------')
            torch.save(model, filename)
            dice_zero = dice
            new_line = '\n' 
            with open(f'./{folder}/log.txt', 'a') as f:
                f.write(f'Split: {SPLIT}, Epoch: {epoch+1}, Dice score: {dice: 0.3f}, Acc: {acc: 0.3f} {new_line}')

            
        
    

if __name__ == '__main__':
    resizes = [256]
    splits = [0]

    for i in splits:
        for j in resizes:
            main(resize = j, SPLIT = i)


