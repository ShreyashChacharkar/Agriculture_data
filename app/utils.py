import numpy as np
import h5py
import warnings
import sklearn
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf 
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, MaxPool2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score
import h5py
from PIL import Image, ImageOps, ImageEnhance
import os
import re
import pandas as pd
import numpy as np
    
    
def load_dataset():
    train_dataset = h5py.File('dataset/rice_data.h5', "r")
    train_set_x_orig = np.array(train_dataset["image_data"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["label"][:]) # your train set labels

    test_dataset = h5py.File('dataset/test_rice_data.h5', "r")
    test_set_x_orig = np.array(test_dataset["image_data"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["label"][:]) # your test set labels

    # classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig

def resized_image(image_path=[], image1=None):
    flat = []
    new_width = 224
    new_height = 224
    if len(image_path) > 0:
        for i in image_path:
            # Open and resize the image using PIL
            image = Image.open(i)
            resized_image = image.resize((new_width, new_height))
            # Convert to a NumPy array and normalize
            img_array = np.array(resized_image) / 255.0
            # Append the processed image to the list
            flat.append(img_array)
    else:
        resized_image = image1.resize((new_width, new_height))
        img_array = np.array(resized_image) / 255.0
        flat.append(img_array)
    processed_images = np.array(flat).reshape(-1, 224, 224, 3)
    return processed_images       

def load_model(model_name):
    load_model = tf.keras.models.load_model(model_name)
    return load_model

def image():
    image = tf.keras.preprocessing.image
    return image

def preprocess_input(x):
    preprocess_input = tf.keras.applications.resnet50.preprocess_input(x)
    return preprocess_input
