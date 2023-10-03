### Script for CS329s ML Deployment Lec 
import streamlit as st
import tensorflow as tf
### from utils import resized_image
import numpy as np
from utils import image, preprocess_input, load_model, resized_image, classes_and_models, update_logger
from PIL import Image
import time

# Setup environment credentials (you'll need to change these)
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "___________________" # change for your GCP key
# PROJECT = "_____________" # change for your GCP project
# REGION = "us-central1" # change for your GCP region (where your model is hosted)

def main():
    ### Streamlit code (works as a straigtht-forward script) ###
    st.title("Welcome to Rice disease Vision 📸")
    st.header("Identify disease to your rice leaves!")


    @st.cache # cache the function so predictions aren't always redone (Streamlit refreshes every click)
    def make_prediction(imagepath, model):
        """
        Takes an image and uses model (a trained TensorFlow model) to make a
        prediction.

        Returns:
        image (preproccessed)
        pred_class (prediction class from class_names)
        pred_conf (model confidence)
        """
        pdict = {0:"Blight",1:"Brown Spot",2:"Leaf Sumt"}
        image_ = Image.open(imagepath)
        pred_x = resized_image(image1=image_)
        prediction = model.predict(pred_x)
        pred = np.argmax(prediction[0])
        return pdict[pred], prediction[0,pred]*100
    

    choose_model = st.sidebar.selectbox(
    "Pick model you'd like to use",
    ("Model 1 (3 diseases)", # original 10 classes
    # "Model 2 (5 classes)", # original 10 classes + donuts
    # "Model 3 (11 food classes + non-food class)"
    ) # 11 classes (same as above) + not_food class
        )

    if choose_model == "Model 1 (3 diseases)":
        CLASSES = classes_and_models["model_1"]["classes"]
        MODEL = classes_and_models["model_1"]["model_name"]
    # elif choose_model == "Model 2 (11 food classes)":
    #     CLASSES = classes_and_models["model_2"]["classes"]
    #     MODEL = classes_and_models["model_2"]["model_name"]
    # else:
    #     CLASSES = classes_and_models["model_3"]["classes"]
    #     MODEL = classes_and_models["model_3"]["model_name"]

    if st.checkbox("Accesible diseases"):
        st.write(f"You chose MODEL which can identify disease of rice: \n", CLASSES)
    # File uploader allows user to add their own image
    uploaded_file = st.file_uploader(label="Upload an image of rice leaves",
                                    type=["png", "jpeg", "jpg"])

    # Usage
    model_file = load_model(MODEL)
    # Create logic for app flow
    if not uploaded_file:
        st.warning("Please upload an image.")
        st.stop()
    else:
        uploaded_image= uploaded_file.read()
        image_placeholder = st.empty()
        image_placeholder.image(uploaded_image, use_column_width=True,)
        pred_button = st.button("Predict")

    result =st.empty()
    fb = st.empty()
    fb_msg = st.empty()
    warning_text = st.empty()
    # Did the user press the predict button?
    if pred_button:
        pred_button = True 

    # And if they did...
    if pred_button:
        pred_class, pred_conf = make_prediction(uploaded_file, model=model_file)
        result.write(f"Prediction: {pred_class}, \
                Confidence: {pred_conf:.3f}")
         
        up_image = Image.open(uploaded_file)
        up_image = resized_image(image1=up_image)
        # # Create feedback mechanism (building a data flywheel)
        feedback = fb.selectbox(
            "Is this correct?",
            ("Select an option", "Yes", "No"))
        if feedback == "Select an option":
            pass
        elif feedback == "Yes":
            fb_msg.write("Thank you for your feedback!")
            # Log prediction information to terminal (this could be stored in Big Query or something...)
            print(update_logger(image=up_image,
                                model_used=MODEL,
                                pred_class=pred_class,
                                pred_conf=pred_conf,
                                correct=True))
        elif feedback == "No":
            correct_class = fb.text_input("What should the correct label be?")
            if correct_class:
                fb_msg.write("Thank you for that, we'll use your help to make our model better!")
                # Log prediction information to terminal (this could be stored in Big Query or something...)
                print(update_logger(image=up_image,
                                    model_used=MODEL,
                                    pred_class=pred_class,
                                    pred_conf=pred_conf,
                                    correct=False,
                                    user_label=correct_class))

        for i in range(60, 0, -1):
            warning_text.write(f"Image will be removed in {i} seconds if no action is taken.")
            time.sleep(1)
        image_placeholder.empty()
        warning_text.empty()
        result.empty()

if __name__ == "__main__":
    main()

