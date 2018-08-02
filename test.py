import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

import cv2
import os
import numpy as np
import keras

import os
os.environ[ " CUDA_VISIBLE_DEVICES " ] = " 0 "

model = Sequential()

def Net_model(nb_classes, lr=0.001,decay=1e-6,momentum=0.9):  
    model.add(Convolution2D(filters=10, kernel_size=(5,5),
                            padding='valid',  
                            input_shape=(448, 448, 3)))  
    model.add(Activation('tanh'))  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
  
    model.add(Convolution2D(filters=20, kernel_size=(10,10)))
    model.add(Activation('tanh'))  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    model.add(Dropout(0.25))  
  
    model.add(Flatten())  
    model.add(Dense(1000))
    model.add(Activation('tanh'))  
    model.add(Dropout(0.5))  
    model.add(Dense(nb_classes))  
    model.add(Activation('softmax'))  
  
    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)  
    model.compile(loss='categorical_crossentropy', optimizer=sgd)  
      
    return model
def convert2label(vector):
    string_array=[]
    for v in vector:
        if v==0:
            string_array.append('柯佳嬿')#柯佳嬿
        if v==1:
            string_array.append('季芹')#季芹
        if v==2:
            string_array.append('孫可芳')#孫可芳
        if v==3:
            string_array.append('桂綸鎂')#桂綸鎂
        if v==4:
            string_array.append('梅芳')#梅芳
        if v==5:
            string_array.append('郭書瑤')#郭書瑤
        if v==6:
            string_array.append('柯素雲')#柯素雲
        if v==7:
            string_array.append('王湘涵')#王湘涵
        if v==8:
            string_array.append('郁方')#郁方
        if v==9:
            string_array.append('溫貞菱')#溫貞菱
    return string_array

def loadImages():
    imageList=[]

    rootdir=r'./test'
    list =os.listdir(rootdir)
    for item in list:
        path=os.path.join(rootdir,item)
        if(os.path.isfile(path)):
            f=cv2.imread(path)
            f=cv2.resize(f, (448, 448))
            imageList.append(f)

    return np.asarray(imageList)

x=loadImages()
x=np.asarray(x)

model=Net_model(nb_classes=10, lr=0.001)
model.load_weights('./star_trained_model_weights.h5')

print(model.predict(x))
print(model.predict_classes(x))
y=convert2label(model.predict_classes(x))
print(y)