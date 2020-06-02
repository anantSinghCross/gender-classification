#_training_code

import pandas as pd
import numpy as np

from tensorflow.contrib.layers import flatten

from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential, Model, load_model
from keras.callbacks import EarlyStopping, Callback
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, ELU,GlobalAveragePooling2D, regularizers
from keras.layers.convolutional import Convolution2D, Cropping2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import adam
from sklearn.utils import shuffle
from keras.utils import np_utils


import time, cv2, glob

global inputShape,size

def kerasModel4():
        model = Sequential()
        model.add(Conv2D(16, (8, 8), strides=(4, 4), padding='valid', input_shape=(sizeW,sizeH,1)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (5, 5), padding="same"))
        model.add(Activation('relu'))
        model.add(GlobalAveragePooling2D())
        # model.add(Dropout(.2))
        # model.add(Activation('relu'))
        # model.add(Dense(1024))
        # model.add(Dropout(.5))
        model.add(Dense(512))
        model.add(Dropout(.1))
        model.add(Activation('relu'))
        # model.add(Dense(256))
        # model.add(Dropout(.5))
        # model.add(Activation('relu'))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        return model

sizeW = 80
sizeH = 110

 ## load Training data : female
potholeTrainImages = glob.glob("C:/Users/anant/Documents/GitHub/gender-classification/dataset/Training/female/*.jpg")
# potholeTrainImages.extend(glob.glob("E:/Major 7sem/pothole-and-plain-rode-images/My Dataset/train/Pothole/*.jpeg"))
# potholeTrainImages.extend(glob.glob("E:/Major 7sem/pothole-and-plain-rode-images/My Dataset/train/Pothole/*.png"))

train1 = [cv2.imread(img,0) for img in potholeTrainImages]
for i in range(0,len(train1)):
    train1[i] = cv2.resize(train1[i],(sizeW,sizeH))
temp1 = np.asarray(train1)


#  ## load Training data : male
nonPotholeTrainImages = glob.glob("C:/Users/anant/Documents/GitHub/gender-classification/dataset/Training/male/*.jpg")
# nonPotholeTrainImages.extend(glob.glob("C:/Users/anant/Desktop/pothole-and-plain-rode-images/My Dataset/train/Plain/*.jpeg"))
# nonPotholeTrainImages.extend(glob.glob("C:/Users/anant/Desktop/pothole-and-plain-rode-images/My Dataset/train/Plain/*.png"))
train2 = [cv2.imread(img,0) for img in nonPotholeTrainImages]
for i in range(0,len(train2)):
    train2[i] = cv2.resize(train2[i],(sizeW,sizeH))
temp2 = np.asarray(train2)


## load Testing data : females
potholeTestImages = glob.glob("C:/Users/anant/Documents/GitHub/gender-classification/dataset/Validation/female/*.jpg")
# nonPotholeTrainImages.extend(glob.glob("C:/Users/anant/Desktop/pothole-and-plain-rode-images/My Dataset/train/Plain/*.jpeg"))
# nonPotholeTrainImages.extend(glob.glob("C:/Users/anant/Desktop/pothole-and-plain-rode-images/My Dataset/train/Plain/*.png"))
test1 = [cv2.imread(img,0) for img in potholeTestImages]
for i in range(0,len(test1)):
    test1[i] = cv2.resize(test1[i],(sizeW,sizeH))
temp3 = np.asarray(test1)


## load Testing data : male
nonPotholeTestImages = glob.glob("C:/Users/anant/Documents/GitHub/gender-classification/dataset/Validation/male/*.jpg")
# nonPotholeTrainImages.extend(glob.glob("C:/Users/anant/Desktop/pothole-and-plain-rode-images/My Dataset/train/Plain/*.jpeg"))
# nonPotholeTrainImages.extend(glob.glob("C:/Users/anant/Desktop/pothole-and-plain-rode-images/My Dataset/train/Plain/*.png"))
test2 = [cv2.imread(img,0) for img in nonPotholeTestImages]
for i in range(0,len(test2)):
    test2[i] = cv2.resize(test2[i],(sizeW,sizeH))
temp4 = np.asarray(test2)


X_train = []
X_train.extend(temp1)
X_train.extend(temp2)
X_train = np.asarray(X_train)


X_test = []
X_test.extend(temp3)
X_test.extend(temp4)
X_test = np.asarray(X_test)


y_train1 = np.zeros([temp1.shape[0]],dtype = int)
y_train2 = np.ones([temp2.shape[0]],dtype = int)
y_test1 = np.zeros([temp3.shape[0]],dtype = int)
y_test2 = np.ones([temp4.shape[0]],dtype = int)

print(y_train1[0])
print(y_train2[0])
print(y_test1[0])
print(y_test2[0])

y_train = []
y_train.extend(y_train1)
y_train.extend(y_train2)
y_train = np.asarray(y_train)

y_test = []
y_test.extend(y_test1)
y_test.extend(y_test2)
y_test = np.asarray(y_test)


X_train,y_train = shuffle(X_train,y_train)
X_test,y_test = shuffle(X_test,y_test)

X_train = X_train.reshape(X_train.shape[0], sizeW, sizeH, 1)
X_test = X_test.reshape(X_test.shape[0], sizeW, sizeH, 1)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


print("train shape X", X_train.shape)
print("train shape y", y_train.shape)

inputShape = (sizeW, sizeH, 1)
model = kerasModel4()

X_train = X_train/255
X_test = X_test/255

model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_train, y_train, epochs=15,validation_split=0.1)

print("")

metricsTrain = model.evaluate(X_train, y_train)
print("Training Accuracy: ",metricsTrain[1]*100,"%")

print("")

metricsTest = model.evaluate(X_test,y_test)
print("Testing Accuracy: ",metricsTest[1]*100,"%")

# print("Saving model weights and configuration file")
model.save('model.h5')
print("Saved model to disk")