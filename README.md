# Distracted Driver Classification for Low-computing Power Devices using Simple CNNs

## Important Notes:

### For mod_6, mod_7, mod_8, mod_9, and mod_10 they are:
	-mod_6 is actually mod_10 in the paper
 	-mod_7 is actually mod_9 in the paper
  	-mod_8 is actually mod_8 in the paper
   	-mod_9 is actually mod_7 in the paper
    -mod_10 is actually mod_6 in the paper

### If you want to skip step 1, you can download the preprocessed dataset files here: https://drive.google.com/drive/folders/10W37JG4QXYmeLFq6sfZXJ3iCuB4Ybk_o?usp=sharing

### 1. Prepare the dataset:

	-First, download the dataset from https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data. We will only use the "train" folder since it has labels.

	-Unzipped the dataset, and put the "train" folder in the "prepare dataset" folder.

	-Run "reformat dataset.py". This will load in the dataset (with the label), resize the images, and convert them to numpy array. The newly formatted dataset will be saved as "training_data.pickle".

	-Then run "divide dataset.py". This will divide the dataset into train data (60%), validation data (20%), and test data (20%). Three files will be saved "new_training_data.pickle", "validation_data.pickle", and "test_data.pickle".



### 2. Train the model:

	-The code for training all of our models is in the folder "training code" folder. Put the "new_training_data.pickle" file in that folder to start training.

### 3. Evaluate the model:
	
	-The codes for evaluating all of our models is in the folder "evalutation\Evaluation Code" folder. Make sure you put the "validation_data.pickle" file in the "Evaluation Code" folder.

	-Note that if you want to load in saved models (that I trained) in the "evaluation" folder, you need Tensorflow version 2.9.1.

	-Or you can retrain the models with whatever version Tensorflow you have, and use that same Tensorflow version to load in your trained models (this takes long time).

----------------------------------------------------------------------------------------------------------------------------------

## RUN MODELs ON RASPBERRY PI 4


### 1. Update the system:
	sudo apt-get update
	sudo apt-get upgrade
   
### 2. Update pip3:
	pip3 install --upgrade pip
   
### 3. Install tensorflow lite:
	pip3 install tflite_runtime==2.9.1

### 4. In the "raspberry pi" folder, there are two folders, "prepare for evaluation" and "evaluation":
	-The "prepare for evaluation" folder contain code that converts the normal models to tflite models.
   
	-The "evaluation" folder contain code to evaluate the proposed model vs the original model on test data. Therefore, copy the "test_data.pickle" and your tflite models to this folder if you want to evaluate the models.
   
