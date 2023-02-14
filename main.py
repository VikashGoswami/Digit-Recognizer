import matplotlib.pyplot as plt
import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from tensorflow.keras.utils import to_categorical
from keras import backend as K
from keras.models import load_model

# from matplotlib.cm import get_cmap
# import numpy
# import pandas as pd


(X_train, y_train), (X_test, y_test) = mnist.load_data()

# plot six samples of mnist


# for i in range(6):
#     plt.subplot(int('23' + str(i+1)))
#     plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

#reshape [samp][wdth][hght][channels]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# covert class vector to binary class

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train = X_train / 255
X_test = X_test / 255


def create_model():

    num_classes = 10
    model = Sequential()
    model.add(Convolution2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
   
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model


# def create_model():
#     num_classes = 10
#     model = Sequential()
#     model.add(Convolution2D(30,(5,5),input_shape = (28,28,1),activation = 'relu'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Convolution2D(15,(3,3),activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Flatten())
#     model.add(Dense(500,activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dropout(0.5))
#     model.add(Dense(num_classes,activation='softmax'))
#     model.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
#     return model

model = create_model()
print("Create model")

#Fit model

hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
print("The model has successfully trained.")


scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

model.save:('hdmodel.h5')
print("The model has successfully saved.")
