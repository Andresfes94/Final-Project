import streamlit as st
from PIL import Image
from classify import predict_VGG16
from classify import predict_color
from classify import predict_category

import pandas as pd
import numpy as np
import streamlit as st
from os import listdir
from os.path import isfile, join
from PIL import Image
#import Training
#import Testing
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.models import load_model
from keras import backend as K
import io
from keras.preprocessing.image import load_img
import streamlit as st
from tempfile import NamedTemporaryFile
import streamlit as st
import numpy as np
from PIL import Image
import os

st.title(" Classification Example")

st.set_option('deprecation.showfileUploaderEncoding', False)
# uploaded_file = st.file_uploader("Choose an image...", type="jpg")
# uploaded_file_ = io.TextIOWrapper(uploaded_file)

st.sidebar.title("About the app")

st.sidebar.info(
    "This is a demo application written to help to visualize the predictions. The application identifies the figure in the picture. It was built using a Convolution Neural Network (CNN)."
    "The app still under construction, but you  will be able to upload your own pictures and train the model with them.")

#onlyfiles = [f for f in listdir('C:/Users/El Matematico/Documents/streamlit_project/StreamlitDemos-master/Classification/Data/Training') if isfile(
   # join('C:/Users/El Matematico/Documents/streamlit_project/StreamlitDemos-master/Classification/Data/Test', f))]
onlyfiles = [ f for f in listdir('C:/Users/El Matematico/Documents/streamlit_project/raw-img/mucca')]
st.sidebar.title("Train Neural Network")
if st.sidebar.button('Train CNN'):
    Training.train()

st.sidebar.title("Predict New Images")
folder_path= 'C:/Users/El Matematico/Documents/streamlit_project/StreamlitDemos-master/Streamlit_Upload/images/'
filenames = os.listdir(folder_path)
selected_filename = st.sidebar.selectbox("Pick an image.", filenames)
if st.sidebar.button('Predict '):
    showpred = 1

    path_file_1=("C:/Users/El Matematico/Documents/streamlit_project/StreamlitDemos-master/Streamlit_Upload/images/" + selected_filename)
    prediction = predict_VGG16(path_file_1)
#st.title('Animal Identification')
st.write("Pick an image from the left. You'll be able to view the image.")
st.write("When you're ready, submit a prediction on the left.")


folder_path= 'C:/Users/El Matematico/Documents/streamlit_project/StreamlitDemos-master/Streamlit_Upload/images/'
filenames = os.listdir(folder_path)
selected_filename = st.selectbox('Select a Picture', filenames)
path_file = os.path.join(folder_path, selected_filename)


#filename = file_selector()
#st.write('You selected `%s`' % path_file)
image = load_img(path_file,target_size=(1600, 1600))
st.image(image, caption="Selected picture", use_column_width=True)
#img_file_buffer = st.file_uploader("Upload an image")
#if img_file_buffer is not None:
 #   image = Image.open(img_file_buffer)
  #  img_array = np.array(image)  # if you want to pass it to OpenCV
    #st.image(image, caption="Selected picture", use_column_width=True)
st.write("")
st.write("")

#st.write("filename:", img_file_buffer.name)
#image = Image.load(image)
st.title("Classifying...")
st.title("Be patient the model is doing its best...")

st.write("")
st.write("")


st.title("Prediction using VGG16:")
label = predict_VGG16(path_file)
st.write('%s (%.2f%%)' % (label[1], label[2] * 100))

st.write("")
st.write("")
st.write("")


st.title("Prediction using Salda19 to get the color:")
label = predict_color(path_file)
st.write("The predicted color is:")
st.title(label)

st.write("")
st.write("")

st.title("Prediction using Salda19 to get the category:")
label = predict_category(path_file)
st.write("The predicted categories are:")
st.title(label)






# buffer = st.file_uploader("Choose an image...", type="jpg")
# temp_file = NamedTemporaryFile(delete=False)
# if buffer:
# temp_file.write(buffer.getvalue())
# st.write(load_img(temp_file.name))
# if uploaded_file_ is not None:
# image = Image.open(buffer)
# st.image(buffer, caption='Uploaded Image.', use_column_width=True)
