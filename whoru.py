import h5py
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,accuracy_score,f1_score,precision_score,recall_score
# suppress display of warnings
import os
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation

st.set_page_config(page_title="WhoRU", page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items={
        'Get Help': 'https://github.com/zwy923/whoRU',
        'Report a bug': "https://github.com/zwy923/whoRU/issues",
        'About': "#This is an *smart system* cool app!"
    })

source_dir="./105_classes_pins_dataset"
@st.cache
def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]
class IdentityMetadata():
    def __init__(self, base, name, file):
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file) 

@st.cache
def load_metadata(path):
    metadata = []
    for i in os.listdir(path):
        for f in os.listdir(os.path.join(path, i)):
            # Check file extension. Allow only jpg/jpeg' files.
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg':
                metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)

# metadata = load_metadata('images')
metadata = load_metadata(source_dir)

@st.cache(allow_output_mutation=True)
def vgg_face():	
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    return model
@st.cache
def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

@st.cache
def show_pair(idx1, idx2):
    plt.figure(figsize=(8,3))
    plt.suptitle(f'Distance between {idx1} & {idx2}= {distance(embeddings[idx1], embeddings[idx2]):.2f}')
    plt.subplot(121)
    plt.imshow(load_image(metadata[idx1].image_path()))
    plt.subplot(122)
    plt.imshow(load_image(metadata[idx2].image_path()))



st.title('Welcome To Smart System Project WhoRU!')
instructions = """
        Either upload your own image or select from
        the sidebar to get a preconfigured image.
        The image you select or upload will be fed
        through the Deep Neural Network in real-time
        and the output will be displayed to the screen.
        """
st.write(instructions)
function_split=["Face recognition","Emotional recognition","Camera capture"]
member=["Wenyue Zhang","Yufeng Pan","Jiaxuan Qi","Guowen Wang","Junyang Huang","Kecen Yin"]
#start
metadata = load_metadata(source_dir)
model = vgg_face()
model.load_weights('./vgg_face_weights.h5')
vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
type(vgg_face_descriptor)
#插入功能









audio_file = open('./assets/sound/The Who - Who Are You.mp3', 'rb')
audio_bytes = audio_file.read()

function_type = st.sidebar.selectbox("Function Select", function_split)
group_member = st.sidebar.selectbox("Group member",member)
with st.sidebar:
    st.audio(audio_bytes, format='audio/ogg')
    st.write("The Who - Who Are You")
st.snow()
if (function_type == "Camera capture"):
    picture = st.camera_input("Take a picture")

elif (function_type == "Emotional recognition"):
    file = st.file_uploader('Upload An Image')
    if(file):
        img = Image.open(file)
        st.success('This is a success img load', icon="✅")
    st.write("looks unhappy?")
    st.snow()
else:
    file = st.file_uploader('Upload An Image')
    if(file):
        img = Image.open(file)
        st.success('This is a success img load', icon="✅")