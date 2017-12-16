from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers.normalization import BatchNormalization
import glob
import csv
import cv2
import os
from six.moves import cPickle as pickle
from numpy import array, asarray, ndarray

batch_size = 5
epochs = 5

img_rows, img_cols = 32, 32

data_root = '.'
with open(os.path.join(data_root, 'data'+str(img_rows)+'x'+str(img_cols)+'.pickle'), 'rb') as f:
    data = pickle.load(f)
    
x_train=asarray(data['x_train'],dtype=float)
x_test=asarray(data['x_test'],dtype=float)
y_train=asarray(data['y_train'],dtype=float)
y_test=asarray(data['y_test'],dtype=float)

num_classes = 100
min_price = min(min(y_train), min(y_test))
class_size = (max(max(y_train), max(y_test)) - min_price)/num_classes
class_size = int(class_size+1)

def convert_to_one_hot(price):
    ans = [0]*num_classes
    ans[int((price-min_price)/class_size)] = 1
    return ans

y_train_one_hot = []
y_test_one_hot = []

for i in y_train:
    y_train_one_hot.append(convert_to_one_hot(i))

for i in y_test:
    y_test_one_hot.append(convert_to_one_hot(i))
    
y_train=asarray(y_train_one_hot,dtype=float)
y_test=asarray(y_test_one_hot,dtype=float)

x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = Sequential()
model.add(Conv2D(16,data_format = 'channels_last', kernel_size=(3, 3),
input_shape=(img_rows, img_cols,3)))         
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Conv2D(32, (2,2)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Conv2D(64, (2,2)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.15))

model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.15))

model.add(Dense(num_classes, activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          shuffle=True)

scores = model.evaluate(x_test, y_test, verbose=0)
print("Error: %.2f%%" % (100-scores[1]*100))

model.save('my_model.h5')
