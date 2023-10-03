import tensorflow as tf 
from PIL import Image
import numpy as np
# from google.cloud import storage

base_classes = ["Blight", "Brown Spot","Leaf smut"]

classes_and_models = {
    "model_1": {
        "classes": base_classes,
        "model_name": "agri-app\model_resnet50V2" 
    }
    # ,
    # "model_2": {
    #     "classes": sorted(base_classes + ["donut"]),
    #     "model_name": "efficientnet_model_2_11_classes"
    # },
    # "model_3": {
    #     "classes": sorted(base_classes + ["donut", "not_food"]),
    #     "model_name": "efficientnet_model_3_12_classes"
    # }
}


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


def update_logger(image, model_used, pred_class, pred_conf, correct=False, user_label=None):
    """
    Function for tracking feedback given in app, updates and reutrns 
    logger dictionary.
    """
    logger = {
        "image": image,
        "model_used": model_used,
        "pred_class": pred_class,
        "pred_conf": pred_conf,
        "correct": correct,
        "user_label": user_label
    }   
    return logger
