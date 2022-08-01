from __future__ import print_function
import keras
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

batch_size=32

# load test set
_, (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype('float32')
x_test /= 255
print(x_test.shape[0], 'test samples')

num_classes = 10

y_test = keras.utils.to_categorical(y_test, num_classes)

import os

# model to load
weights_path = 'model.46-0.42709.hdf5'
out_dir = 'visual-features'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
model = Sequential()

# Test

model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_test.shape[1:], name='conv1'))
model.add(BatchNormalization(axis=3, name='bn_conv1'))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), name='conv2'))
model.add(BatchNormalization(axis=3, name='bn_conv2'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', name='conv3'))
model.add(BatchNormalization(axis=3, name='bn_conv3'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), name='conv4'))
model.add(BatchNormalization(axis=3, name='bn_conv4'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, name='fc1'))
model.add(BatchNormalization(axis=1, name='bn_fc1'))
model.add(Activation('relu'))
model.add(Dense(num_classes, name='output'))
model.add(BatchNormalization(axis=1, name='bn_outptut'))
model.add(Activation('softmax'))

model.load_weights(weights_path)

# Extract features and save them

feat_extractor = Model(inputs=model.input,
                       outputs=model.get_layer('fc1').output)

features = feat_extractor.predict(x_test, batch_size=batch_size)

np.save(os.path.join(out_dir, 'fc1_features.npy'), features)