import streamlit as st
import os
import requests
from PIL import Image
import pickle
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

def download_file(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as file:
        file.write(response.content)

# Download the "features.pkl" file from the external source
file_id = '1BJqWitd8T1147XihhcOEWhhRLLXzoQHf'
file_url = f'https://drive.google.com/uc?id={file_id}'
download_path = 'features.pkl'
download_file(file_url, download_path)
with open(download_path, 'rb') as file:
    features_data = file.read()
feature_list = np.array(pickle.load(io.BytesIO(features_data)))

filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
st.title('Fashion Recommendation System')

def saveFile(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def featureExtract(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])
    return indices

uploaded_file = st.file_uploader("Upload an image")
if uploaded_file is not None:
    if saveFile(uploaded_file):
        display_img = Image.open(uploaded_file)
        st.image(display_img)
        features = featureExtract(os.path.join("uploads", uploaded_file.name), model)
        
        indices = recommend(features, feature_list)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header("Some error occured in file upload")
