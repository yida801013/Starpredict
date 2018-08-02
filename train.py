
# coding: utf-8




import os

os.environ[ " CUDA_VISIBLE_DEVICES " ] = " 0 "
import cv2
print(cv2.__version__)
from keras.utils import to_categorical
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

#GPU


import keras




import time
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

import os
import numpy as np
import cv2
start = time.time()
model = Sequential()

def loadImages():
    imageList=[]
    labelList=[]

    rootdir= r'./star/JIAYAN_face/'
    list =os.listdir(rootdir)
    for item in list:
        path=os.path.join(rootdir,item)
        if(os.path.isfile(path)):
            f=cv2.imread(path)
            f=cv2.resize(f, (448, 448))#resize到網絡input的shape
            imageList.append(f)
            labelList.append(0)#類別0
            
    rootdir= r'./star/JICIN_face/'
    list =os.listdir(rootdir)
    for item in list:
        path=os.path.join(rootdir,item)
        if(os.path.isfile(path)):
            f=cv2.imread(path)
            f=cv2.resize(f, (448, 448))#resize到網絡input的shape
            imageList.append(f)
            labelList.append(1)#類別1
            
    rootdir= r'./star/KEFANG_face/'
    list =os.listdir(rootdir)
    for item in list:
        path=os.path.join(rootdir,item)
        if(os.path.isfile(path)):
            f=cv2.imread(path)
            f=cv2.resize(f, (448, 448))#resize到網絡input的shape
            imageList.append(f)
            labelList.append(2)#類別2
            
    rootdir= r'./star/LUNMEI_face/'
    list =os.listdir(rootdir)
    for item in list:
        path=os.path.join(rootdir,item)
        if(os.path.isfile(path)):
            f=cv2.imread(path)
            f=cv2.resize(f, (448, 448))#resize到網絡input的shape
            imageList.append(f)
            labelList.append(3)#類別3
            
    rootdir= r'./star/MEIFANG_face/'
    list =os.listdir(rootdir)
    for item in list:
        path=os.path.join(rootdir,item)
        if(os.path.isfile(path)):
            f=cv2.imread(path)
            f=cv2.resize(f, (448, 448))#resize到網絡input的shape
            imageList.append(f)
            labelList.append(4)#類別4
            
    rootdir= r'./star/SHUYAO_face/'
    list =os.listdir(rootdir)
    for item in list:
        path=os.path.join(rootdir,item)
        if(os.path.isfile(path)):
            f=cv2.imread(path)
            f=cv2.resize(f, (448, 448))#resize到網絡input的shape
            imageList.append(f)
            labelList.append(5)#類別5
            
    rootdir= r'./star/SUYUN_face/'
    list =os.listdir(rootdir)
    for item in list:
        path=os.path.join(rootdir,item)
        if(os.path.isfile(path)):
            f=cv2.imread(path)
            f=cv2.resize(f, (448, 448))#resize到網絡input的shape
            imageList.append(f)
            labelList.append(6)#類別6
            
    rootdir= r'./star/XIANGHAN_face/'
    list =os.listdir(rootdir)
    for item in list:
        path=os.path.join(rootdir,item)
        if(os.path.isfile(path)):
            f=cv2.imread(path)
            f=cv2.resize(f, (448, 448))#resize到網絡input的shape
            imageList.append(f)
            labelList.append(7)#類別7
            
    rootdir= r'./star/YUFANG_face/'
    list =os.listdir(rootdir)
    for item in list:
        path=os.path.join(rootdir,item)
        if(os.path.isfile(path)):
            f=cv2.imread(path)
            f=cv2.resize(f, (448, 448))#resize到網絡input的shape
            imageList.append(f)
            labelList.append(8)#類別8
            
    rootdir= r'./star/ZHENLING_face/'
    list =os.listdir(rootdir)
    for item in list:
        path=os.path.join(rootdir,item)
        if(os.path.isfile(path)):
            f=cv2.imread(path)
            f=cv2.resize(f, (448, 448))#resize到網絡input的shape
            imageList.append(f)
            labelList.append(9)#類別9

   

    return np.asarray(imageList), to_categorical(labelList, 10)


def Net_model(nb_classes, lr=0.001,decay=1e-6,momentum=0.9):
    model.add(Convolution2D(filters=10, kernel_size=(5,5),
                            padding='valid',
                            input_shape=(448, 448, 3)))  #卷積層
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  #池化層
    model.add(Convolution2D(filters=20, kernel_size=(10,10)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten()) #平坦層
    model.add(Dense(1000))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    return model

nb_classes = 10
nb_epoch = 30
nb_step = 6
batch_size = 64

x,y = loadImages()

from keras.preprocessing.image import ImageDataGenerator
dataGenerator = ImageDataGenerator()
dataGenerator.fit(x)
data_generator = dataGenerator.flow(x, y, batch_size, True) #generator函數，用來生成批處理數據

model = Net_model(nb_classes=nb_classes, lr=0.0001) #加載網絡模型

history = model.fit_generator(data_generator, epochs=nb_epoch, steps_per_epoch = nb_step, shuffle = True) #訓練網絡

model.save_weights('./star_trained_model_weights.h5') #將圖片處理成h5格式
print("DONE, model saved in path")

end = time.time()
print(end-start)


# In[ ]:


import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.show()

