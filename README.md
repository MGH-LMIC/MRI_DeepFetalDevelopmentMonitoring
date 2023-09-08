# Deep Fetal Development Monitoring (DFDM)
## Overview
The Deep Fetal Development Monitoring project leverages deep learning techniques to quantify fetal fluid volume and fetal weight through MRI image segmentation. This repository contains the code and instructions to preprocess, train, test, and predict volumes using the trained model.Please note that no data is added to this repository to protect patient privacy. You can you your own data for training and testing. 

## Getting Started
To get started with the project, follow these steps:

## Data Preparation:

Add your DICOM image files to the dicoms folder ```./dicoms```.
Add corresponding mask images to the masks folder ```./masks```.
The config.txt contains the names of directories where data is stored. Make sure to create "dicoms" and "masks" folders before starting data processing.
Save dicom files in the "dicoms" folder and mask dicom files in the "masks" folder. Please make sure that all code files and data folders are in the same parent directory.
## Data Preprocessing:

Run preprocess.py to preprocess the data. This step prepares the data for training and testing.
Preprocessing is done in the following steps:
1- Moving all dicom files to the parent directory of each case folder.
2- Converting all dicom files of images and masks to png image format and saving them in new directories ```./pngs```, ```./png_masks```.
3- Processing image files and mask files and saving them to new directories ```./processed_images```, ```./processed_masks```.
Once run, the preprocess.py will create all the necessary directories based on the config.txt file.
```
python preprocess.py
```
## Data Splitting:

Execute split_data.py to split the dataset into training, validation, and testing sets.
Data splitting is done according to the following steps:
1- The counts of train, val, and test sets are calculated as 80:10:10 proportions of the data, respectivley.
2- Images and masks from all cases are moved to mixed folders ```./mixed_images```, ```./mixed_masks```.
3- A split folder is created to combine the splitted data ```./split_0```.
3- Files are copied to corresponding directories e.g. ```./split_0/train_images```, ```./split_0/train_masks``` , ...etc.
```
python split_data.py
```
## Model Training:

Train your deep learning model by running train.py. This step uses the prepared dataset to train the model for fetal fluid volume and fetal body segmentation.
The training process is run for 100 epochs with each epoch's Dice score is recorded in a log fole ```./log.txt```.
Model is saved when Dice score is greater than the previous epoch's. The model file is saved in the same parent directory ```./model_256_split_0.pt```.
```
python train.py
```
## Model Testing:

After training, evaluate the model's performance with the testing dataset using test.py.
The test.py script creates dataloaders from the test folders and loads the trained model and then calculates the total Dice score of the test set. All test output is recorded in the ```./test_log.txt``` file.
```
python test.py
```
## Inference:

Use the trained model to make predictions on new data by running predict_volumes.py.
The file all_new_afv_volums_small.csv contains the original dicom image sizes which are required for rescaling predicted masks to the original size before volume calculation.
Volume calculation is done according to the following steps:
1- Calculating the ground truth volumes using the ground truth masks for each case and saving them in a dictionary. 
2- Claculating the predicted volumes using the trained model for each case and saving them in a dictionary. 
3- Combining the saved dictionaries in one dataframe and exporting it as a csv file ```./fetal_vol_GT_AI.csv```
```
python predict_volumes.py
```
## Requirements
Python (3.x)
PyTorch 
## License
This project is licensed under the MIT License.
