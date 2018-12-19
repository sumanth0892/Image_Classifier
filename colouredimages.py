from keras.datasets import cifar10
from keras import models,layers
(train_images,train_labels),(test_images,test_labels) = cifar10.load_data()
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.summary()
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.summary()

x_train = train_images.reshape((50000,32,32,3))
x_train = x_train.astype('float32')/255
x_test = test_images.reshape((50000,32,32,3))
x_test = x_test.astype('float32')/255

from keras.utils import to_categorical
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',
              metrics=['accuracy','mae'])
history = model.fit(x_train,y_train,epochs=5,batch_size=64)
test_loss,test_acc = model.evaluate(x_test,y_test)
print(history.history['loss'])
print(history.history['acc'])
print(test_loss)
print(test_acc)


