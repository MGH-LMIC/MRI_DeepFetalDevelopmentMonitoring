import torch
import torchvision
from dataset import MRIDataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
# from utils import *
import configparser
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from datetime import datetime

# Get the current date and time
now = datetime.now()

# Format the date and time
formatted_date = now.strftime("%d, %B, %Y, %H:%M")

# Create a ConfigParser object
config = configparser.ConfigParser()

# Read the external text file
config.read('config.txt')


folder = config.get('variables', 'folder')

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

def boundary_points(mask):
    """
    Get the boundary points from a binary mask.
    """
    contour = np.zeros_like(mask, dtype=bool)
    mask = mask.astype(bool)  # Convert to boolean type
    contour[1:-1, 1:-1] = mask[1:-1, 1:-1] & ~mask[:-2, 1:-1] & ~mask[2:, 1:-1] & ~mask[1:-1, :-2] & ~mask[1:-1, 2:]
    return np.array(np.nonzero(contour))

def hausdorff_distance(mask1, mask2):
    """
    Calculate the Hausdorff distance between two binary masks.
    """
    points1 = boundary_points(mask1)
    points2 = boundary_points(mask2)

    if points1.size == 0 or points2.size == 0:
        # If one of the masks has no boundary points, return a default value (e.g., -1)
        return -1

    dist_1_to_2 = np.max(np.min(np.linalg.norm(points1[:, np.newaxis] - points2, axis=-1), axis=0))
    dist_2_to_1 = np.max(np.min(np.linalg.norm(points2[:, np.newaxis] - points1, axis=-1), axis=0))

    return max(dist_1_to_2, dist_2_to_1)


def calculate_precision_recall(predictions, true_masks, threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    for pred_mask, true_mask in zip(predictions, true_masks):
        # Binarize predicted and true masks using threshold
        pred_mask = (pred_mask > threshold).cpu().numpy().astype(int)
        true_mask = (true_mask > threshold).cpu().numpy().astype(int)

        # Calculate true positives, false positives, and false negatives
        true_positives += ((pred_mask == 1) & (true_mask == 1)).sum()
        false_positives += ((pred_mask == 1) & (true_mask == 0)).sum()
        false_negatives += ((pred_mask == 0) & (true_mask == 1)).sum()
        true_negatives += ((pred_mask == 0) & (true_mask == 0)).sum()

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    sensitivity = true_positives / (true_positives + false_negatives)
    specificity = true_negatives/ (true_negatives + false_positives)

    return precision, recall, sensitivity, specificity

def check_accuracy(loader, model, device = 'cuda'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    hs = 0
    model.eval()
    true_mask = []
    pred_mask = []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2*(preds*y).sum()) / ((preds + y).sum() + 1e-8)
            true_mask.append(y)
            pred_mask.append(preds)

            
        print(f'Got {num_correct}/{num_pixels} with acc {num_correct/ num_pixels*100 : 0.2f}')
        print(f'Dice score: {dice_score/len(loader) : 0.3f}')
    return dice_score/len(loader) , num_correct/ (num_pixels*100 + 1e-8), true_mask,pred_mask
    



DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'

model_path = './model_256_split_0.pt'
# model_path = f'./{folder}/model_160_split_0.pt'
model = torch.load(model_path).to(DEVICE)

batch_size = 4
num_workers = 2
pin_memory = True

for n in range(1):

	TEST_IMG_DIR = f'./{folder}/split_{n}/test_images/'
	TEST_MASK_DIR = f'./{folder}/split_{n}/test_masks/'

	test_ds = MRIDataset(image_dir = TEST_IMG_DIR,mask_dir = TEST_MASK_DIR,transform = val_transform)
	    
	test_loader = DataLoader(test_ds,batch_size = batch_size,num_workers = num_workers,pin_memory = pin_memory,shuffle = False)


	dice, acc, true_mask, pred_mask = check_accuracy(test_loader, model, device = DEVICE)

	# precision, recall, sensitivity, specificity = calculate_precision_recall(pred_mask, true_mask , 0.5)




	new_line = '\n' 
	with open(f'./{folder}/test_log.txt', 'a') as f:
		f.write(f'Dice score: --------({dice: 0.3f} )------- , Model: {model_path.split("/")[-2]}, Date: {formatted_date} {new_line}')
		