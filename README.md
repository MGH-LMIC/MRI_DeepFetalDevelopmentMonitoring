# MRI_DeepFetalDevelopmentMonitoring
## Deep Fetal Development Monitoring
##Overview
The Deep Fetal Development Monitoring project leverages deep learning techniques to quantify fetal fluid volume and fetal weight through image segmentation. This repository contains the code and instructions to preprocess, train, test, and predict volumes using the trained model.

##Getting Started
To get started with the project, follow these steps:

##Data Preparation:

Add your DICOM image files to the dicoms folder.
Add corresponding mask images to the masks folder.
##Data Preprocessing:

Run preprocess.py to preprocess the data. This step prepares the data for training and testing.
##Data Splitting:

Execute split_data.py to split the dataset into training, validation, and testing sets.
##Model Training:

Train your deep learning model by running train.py. This step uses the prepared dataset to train the model for fetal fluid volume and fetal weight segmentation.
##Model Testing:

After training, evaluate the model's performance with the testing dataset using test.py.
#Inference:

Use the trained model to make predictions on new data by running predict_volumes.py.
##Requirements
Python (3.x)
PyTorch 
##License
This project is licensed under the MIT License.
