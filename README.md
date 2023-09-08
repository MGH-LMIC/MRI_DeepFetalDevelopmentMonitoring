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
Once run, the preprocess.py will create all the necessary directories based on the config.txt file.
```
python preprocess.py
```
## Data Splitting:

Execute split_data.py to split the dataset into training, validation, and testing sets.
```
python split_data.py
```
## Model Training:

Train your deep learning model by running train.py. This step uses the prepared dataset to train the model for fetal fluid volume and fetal body segmentation.
```
python train.py
```
## Model Testing:

After training, evaluate the model's performance with the testing dataset using test.py.
```
python test.py
```
## Inference:

Use the trained model to make predictions on new data by running predict_volumes.py.
The file all_new_afv_volums_small.csv contains the original dicom image sizes which are required for rescaling predicted masks to the original size before volume calculation.
```
python predict_volumes.py
```
## Requirements
Python (3.x)
PyTorch 
## License
This project is licensed under the MIT License.
