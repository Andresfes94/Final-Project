import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import io
from keras.preprocessing.image import load_img
import tensorflow as tf
import numpy as np
#def predict(image1):
#    model = VGG16()
#    image = load_img(image1, target_size=(224, 224))
#    # convert the image pixels to a numpy array
#    image = img_to_array(image)
    # reshape data for the model
#    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
#    image = preprocess_input(image)
    # predict the probability across all output classes
#    yhat = model.predict(image)
    # convert the probabilities to class labels
#    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
 #   label = label[0][0]
   # return label



def predict_VGG16(image1):
    #path = 'C:/Users/El Matematico/Documents/streamlit_project/StreamlitDemos-master/Streamlit_Upload/images/'
    model = VGG16()
    image = load_img(image1,target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    # return highest probability
    label = label[0][0]
    return label

def predict_color(image1):
    #path = 'C:/Users/El Matematico/Documents/streamlit_project/StreamlitDemos-master/Streamlit_Upload/images/'
    #model = VGG16()
    model = keras.models.load_model('cnn_color_model.h5py')
    #model = keras.models.load_model('cnn_folder')

    image = load_img(image1,target_size=(64, 64))
    image = img_to_array(image)
    image = tf.expand_dims(image, 0)
    #image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    #image = preprocess_input(image)
    yhat = model.predict(image)
    yhat = yhat.tolist()
    #top_indices = np.argsort(yhat)[0, ::-1][:5]
    class_indices = [(0, 'black'), (1, 'blue'), (2, 'brown'), (3, 'green'), (4, 'grey'), (5, 'navy_blue'), (6, 'pink'),
                     (7, 'purple'), (8, 'red'), (9, 'silver'), (10, 'white'), (11, 'yellow')]
    for i in yhat[0]:
        if i == 1:
            colores=[]
            index = yhat[0].index(i)
            color = class_indices[index][1]
            colores.append(color)
    return colores[0:1]

    #class_indices ={'black': 0, 'blue': 1, 'brown': 2, 'green': 3, 'grey': 4, 'navy_blue': 5, 'pink': 6, 'purple': 7, 'red': 8, 'silver': 9, 'white': 10, 'yellow': 11}


def predict_category(image1):
    #path = 'C:/Users/El Matematico/Documents/streamlit_project/StreamlitDemos-master/Streamlit_Upload/images/'
    #model = VGG16()
    model = keras.models.load_model('cnn_categories_model.h5py')


    image = load_img(image1,target_size=(64, 64))
    image = img_to_array(image)
    image = tf.expand_dims(image, 0)
    #image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    #image = preprocess_input(image)
    yhat = model.predict(image)
    yhat = np.argsort(yhat)[0, ::-1][:5]
    yhat = yhat.tolist()
    #top_indices = np.argsort(yhat)[0, ::-1][:5]
    class_indices = [(0,'Tshirts'), (1,'casual_shoes'), (2,'dogs'), (3,'shirts'), (4,'sport_shoes'), (5,'watches')]
    categories = [class_indices[yhat[0]][1]]

    #for i in yhat:
    #    if i == 1:
       #     index = yhat.index(i)


    return categories