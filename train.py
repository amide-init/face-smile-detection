from typing_extensions import Required
import imutils
from keras.backend import argmax
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils import img_to_array
from keras import utils
from nn.conv import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import os
import cv2
# python train.py --dataset ./datasets/smileD  --model ./output/lenet.hdf5
ap =  argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to image dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

data = []
labels = []

for imagePath in sorted(list(paths.list_images(args["dataset"]))):
    image = cv2.imread(imagePath) #(H, W, Depth = 3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#(H, W, Depth = 1)
    image = imutils.resize(image, width=28)
    image = img_to_array(image)
    data.append(image)

    label = imagePath.split(os.path.sep)[-2]
    label = "smiling" if label == "1" else "not_smiling"
    labels.append(label) 

data =  np.array(data, dtype="float")/255.0
labels = np.array(labels)

le = LabelEncoder().fit(labels)
# 0 : [1,0]
# 1 : [0,1]
labels = utils.to_categorical(le.transform(labels), 2)

classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals
type(classWeight)

(trainX, testX, trainY, testY) =  train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)
# print(trainX.shape)
# print(testX.shape)
# print(trainY[0])
print("compiling model")
model = LeNet.build(width=28, height=28, depth=1, classes=2)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


print("training network ...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
           batch_size=64, epochs=15, verbose=1)

print("evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))


print("save model")
model.save(args['model'])
