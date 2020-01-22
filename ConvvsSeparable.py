#Convolution vs Depthwise Separable Convolution
import os
import numpy as np
from keras.datasets import mnist
from keras import models,layers
from keras.utils import to_categorical
(X,Y),(x,y) = mnist.load_data() #Training and testing data with labels

X = X.reshape((60000,28,28,1))
X = X.astype('float32')/255
x = x.reshape((10000,28,28,1))
x = x.astype('float32')/255
Y = to_categorical(Y); y = to_categorical(y)

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['acc'])
history = model.fit(X,Y,epochs = 10,batch_size = 128,verbose = 1)

test_loss,test_acc = model.evaluate(x,y)
print("Test loss with Convolution layer is: ")
print(test_loss)
print("Test accuracy with Convolution layer is: ")
print(test_acc)


loss = history.history['loss']
acc = history.history['acc']
epochs = range(len(loss))

import matplotlib.pyplot as plt
plt.plot(epochs,loss,'bo',label = 'Training Loss')
plt.plot(epochs,acc,'b',label = 'Training accuracy')
plt.title('Training and Validation losses')
plt.legend()
plt.show()

"""
#Separable Convolution layers
model = models.Sequential()
model.add(layers.SeparableConv2D(32,3,activation='relu',
                                 input_shape=(28,28,1)))
model.add(layers.MaxPooling2D(2))
model.add(layers.SeparableConv2D(64,3,activation='relu'))
model.add(layers.MaxPooling2D(2))
model.add(layers.SeparableConv2D(64,3,activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['acc'])
history = model.fit(X,Y,epochs = 10,batch_size = 128,verbose = 1)
test_loss,test_acc = model.evaluate(x,y)
print("Test loss with Convolution layer is: ")
print(test_loss)
print("Test accuracy with Convolution layer is: ")
print(test_acc)
loss = history.history['loss']
acc = history.history['acc']
epochs = range(len(loss))

test_loss,test_acc = model.evaluate(x,y)
print("Test loss with Convolution layer is: ")
print(test_loss)
print("Test accuracy with Convolution layer is: ")
print(test_acc)

import matplotlib.pyplot as plt
plt.plot(epochs,loss,'bo',label = 'Training Loss')
plt.plot(epochs,acc,'b',label = 'Training accuracy')
plt.title('Training and Validation losses')
plt.legend()
plt.show()
"""


