import h5py
import imageio
import streamlit as st
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,accuracy_score,f1_score,precision_score,recall_score
# suppress display of warnings
import os
import cv2
from keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import tensorflow as tf
import requests
from bs4 import BeautifulSoup
from lxml import html


st.set_page_config(page_title="WhoRU", page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

source_dir="./105_classes_pins_dataset"
@st.cache
def load_image(path):
    img = cv2.imread(path, 1)
    img = cv2.resize(img, dsize = (224,224))
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


def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))


@st.cache
def load_audio():
    audio_file = open('./assets/sound/The Who - Who Are You.mp3', 'rb')
    audio_bytes = audio_file.read()
    return audio_bytes

st.title('Welcome To Smart System Project WhoRU!')
instructions = """
        Select a function from the sidebar to upload your own image. The image you select or upload will be input into the depth neural network in real time, and the output will be displayed on the screen.
        """
st.write(instructions)
function_split=["Face recognition","Emotional recognition","Camera capture"]
member=["Wenyue Zhang","Yufeng Pan","Jiaxuan Qi","Guowen Wang","Junyang Huang","Kecen Yin"]
#start


metadata = load_metadata(source_dir)
model = vgg_face()
model.load_weights('./vgg_face_weights.h5')
vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
embeddings = np.load('data.npy')


@st.cache
def extract_face_embeddings(img_):
    # Load and preprocess the image
    
    img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    img_ = (img_ / 255.).astype(np.float32)
    img_ = cv2.resize(img_, dsize=(224, 224))

    # Generate the embedding vector
    embedding_vector = vgg_face_descriptor.predict(np.expand_dims(img_, axis=0))[0]

    return embedding_vector

def get_wikipedia_data(name):
  # 将名字转化为维基百科可识别的格式，例如将空格替换为'_'
  name = name.replace(' ', '_')
  
  # 构造维基百科的URL
  url = f'https://en.wikipedia.org/wiki/{name}'
  
  # 使用requests库的get函数发送HTTP GET请求
  response = requests.get(url)
  
  # 如果返回的状态码不是200，则请求失败
  if response.status_code != 200:
    return None
  
  # 返回响应内容
  return response.text


def parse_wikipedia_data(text):
  # 使用lxml的html模块解析HTML
  tree = html.fromstring(text)
  
  # 使用XPath表达式查找元素
  element = tree.xpath('//*[@id="mw-content-text"]/div[1]/p[2]')
  print(html.tostring(element[0], method='text', encoding='unicode'))
  # 返回元素的文本内容
  return html.tostring(element[0], method='text', encoding='unicode')
  

def train(img_path,img):
    cv2.imwrite(img_path, img)
    with open(img_path, 'rb') as f:
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    embedding0=extract_face_embeddings(img)
    train_idx = np.arange(metadata.shape[0]) % 9 != 0
    X_train = embeddings[train_idx]
    scaler = StandardScaler()
    X_test = embedding0
    X_test = X_test.reshape(1, -1)
    targets = np.array([m.name for m in metadata])
        
    y_train = targets[train_idx]

    y_test = targets[0]
    y_test = np.array(y_test).reshape(-1)

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    y_test_encoded = le.transform(y_test)

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test.reshape(-1,2622 ))

    pca = PCA(n_components=128)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)


    clf = SVC(C=5., gamma=0.001)
    clf.fit(X_train_pca, y_train_encoded)
    y_predict = clf.predict(X_test_pca)

    y_predict_encoded = le.inverse_transform(y_predict)

    example_image = load_image(img_path)
    example_prediction = y_predict[0]
    example_identity =  y_predict_encoded[0]

    name=example_identity[5:]
    
    st.subheader(f'Identified as {name}')
    st.image(img,width=300)
    if (name=="Wenyue Zhang"):
        st.write("Meet Wenyue, a 21-year-old computer science major hailing from China who speaks English fluently. Zhang Wenyue is also a fun-loving individual with a great sense of humor. In his free time, he enjoys coding up a storm and playing video games with his friends. He's also an avid fan of science fiction and loves to immerse himself in futuristic worlds through books, movies. But above all, Wenyue is a kind and compassionate person who is always willing to lend a helping hand to those in need. So next time you see him around campus, be sure to say hi and get to know this unique and talented individual.")
    else:
        html = get_wikipedia_data(name)
        if html:
            data = parse_wikipedia_data(html)
            st.write(data)  
        else:
            print('Request failed')
def emotion(img):
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}   
    model = load_model('./63.h5',compile=False)
    x = img
    x = cv2.resize(x, (48, 48), interpolation=cv2.INTER_CUBIC)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = np.expand_dims(x, axis=-1).astype(np.float32) / 255.0
    prediction = model.predict(np.expand_dims(x, axis=0))
    predicted_label = np.argmax(prediction, axis=-1)
    predicted_label = int(predicted_label)
    predicted_emotion = emotion_dict[predicted_label]
    
    return predicted_emotion


#插入功能
audio_bytes = load_audio()
function_type = st.sidebar.selectbox("Function Select", function_split)
group_member = st.sidebar.selectbox("Group member",member)
with st.sidebar:
    st.audio(audio_bytes, format='audio/ogg')
    st.write("The Who - Who Are You")
if (function_type == "Camera capture"):
    picture = st.camera_input("Take a picture")
    if(picture):
        image = Image.open(picture)
        img_array = np.array(image)
        img_path ="./"+picture.name
        img = imageio.imread(picture)
        train(img_path,img)
        predicted_emotion=emotion(img_array)
        st.subheader(f"Predicted emotion: {predicted_emotion}")

elif (function_type == "Emotional recognition"):
    st.image("./assets/img/output2.png")
    st.write("Experience the power of AI-powered emotion recognition - never second-guess how someone is feeling again!")
    file = st.file_uploader('Upload An Image')
    if(file):
        img = Image.open(file)
        img_array = np.array(img)
        st.success('This is a success img load', icon="✅")
        predicted_emotion=emotion(img_array)
        st.subheader(f"Predicted emotion: {predicted_emotion}")
        st.image(img,width=300)
else:
    st.snow()
    st.image("./assets/img/output1.png")
    st.write("\nIdentify any celebrity with just a photo - our AI-powered celebrity recognition app does the work for you!\n")
    file = st.file_uploader('Upload An Image')
    if(file):
        img = Image.open(file)
        st.success('This is a success img load', icon="✅")
        img_path ="./"+file.name
        img = imageio.imread(file)
        train(img_path,img)