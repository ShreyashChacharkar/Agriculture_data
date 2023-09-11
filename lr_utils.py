import numpy as np
import h5py
    
    
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

