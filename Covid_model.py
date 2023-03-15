## import packages
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import shutil
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import argparse

from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, save_model, Sequential
from tensorflow.keras.applications import ResNet50V2, MobileNetV3Large
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam

def preprocess():
    ## data preprocessing
    # 1) Load training set
    imagePath = "./all/train"
    imagePaths = list(paths.list_images(imagePath))     # loads only image files from path

    # 2) Resize 
    data = []
    labels = []
    for imagefile in imagePaths:
        label = imagefile.split(os.path.sep)[-2]
        image = cv2.imread(imagefile)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        data.append(image)
        labels.append(label)

    # 3) Normalization 
    data = np.array(data) / 255.0

    # 4) One-hot encoding
    labels = np.array(labels)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    labels = to_categorical(integer_encoded)

    # 5) Dividing training sets
    (x_train, x_val, y_train, y_val) = train_test_split(data, labels, test_size=0.20, stratify=labels)
    
    return x_train, x_val, y_train, y_val

def train_and_save(x_train, x_val, y_train, y_val, save_path, epochs):
    ## AI model training
    model = Sequential()
    adam_s = Adam(learning_rate = 0.00001)

    # 1) Load backbone model : InceptionV3 or MobileNetV3Small
    model.add(backbone_model(input_shape=(256, 256, 3),include_top=False, weights='imagenet',pooling='average'))

    for layer in model.layers:
        layer.trainable = False

    # 2) Add classifier  
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    # 3) Compile and visualize our model
    model.compile(loss='categorical_crossentropy', optimizer=adam_s, metrics=['accuracy'])
    model.summary()

    # training AI model
    H = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val), batch_size=8)

    # save AI model
    save_model(model, save_path)
    return H
    
def save_learning_curve(H, epochs):
    ## Visualization (Optional)
    plt.style.use("ggplot")
    plt.figure()

    plot_dictionary = {"loss": "train_loss", "val_loss": "val_loss", "accuracy": "train_acc", "val_accuracy": "val_acc"}
    for key, label in plot_dictionary.items():
        plt.plot(np.arange(0, epochs), H.history[key], label=label)

    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epochs")
    plt.legend(loc="right")
    plt.savefig("./training_log_plot_"+args.model+".png")

model_dict = {"MobileNetV3Large":MobileNetV3Large, "InceptionV3":InceptionV3, "ResNet50V2":ResNet50V2}
backbone_model = MobileNetV3Large
version = '1'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Determine backbone model")
    parser.add_argument('-m', '--model', required=True, help='input the name of backbone model : MobileNetV3Large, InceptionV3, ResNet50V2')
    parser.add_argument('-v', '--version', type=str, default=version, required=False, help='version number')
    parser.add_argument('--epochs', type=int, default=10, required=False, help='number of epochs')
    
    args, _ = parser.parse_known_args()
    
    try:
        backbone_model = model_dict[args.model]
    except ValueError as ve:
        print('Not a valid model name.')
        raise ve
        
    version = args.version
    epochs = args.epochs
    
    save_path = os.path.join("./covid-"+version, "1")
    logfile = os.path.join(save_path, 'log.txt')
    
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
               
    x_train, x_val, y_train, y_val = preprocess()
    H = train_and_save(x_train, x_val, y_train, y_val, save_path, epochs)
    save_learning_curve(H, epochs)
    