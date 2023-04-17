PROJECT REPORT: https://github.com/Adriman32/RNN-Image-Captions/tree/main/assets/Report.pdf
# RNN Image Captions
Generates image captions using Recurrent Neural Networks.

Implementation of https://towardsdatascience.com/image-captioning-with-keras-teaching-computers-to-describe-pictures-c88a46a311b8

### Description
- Image encoding using InceptionV3 Convolutional Neural Network
- LSTM-based Recurrent Neural Network for predicting word sequences from partial captions
- Generates captions using Greedy Search for selecting most likely next word

# Requirements
- Python
- requirements.txt (included)
- Flickr-8K Dataset


# How To Use
In your console, navigate to desired directory and enter the following
- git clone https://github.com/Adriman32/RNN-Image-Captions.git
- cd RNN-Image-Captions
- pip install -r requirements.txt


Running ***main.py*** will begin the experiment. Given the appropriate filepaths, the code will create dicts with image names and pre-labeled captions as key-value pairs. These dicts will be split into separate train/test sets for use in training and evaluation. 

It is recommended to pre-compile the image encodings, which is assumed in the code. ***encoded_images.pkl*** is included in the project, and the code contains functions for converting the images into their (2048,) encodings if desired.


### Pre-Trained Models
The `models` folder included in the repository contains several different models with varying hyperparameters. Currently, the following models are available:

*model_epoch_19.h5*: Model trained on basic configuration for 20 epochs.

![Figure 0](/assets/Figure_0.png)





