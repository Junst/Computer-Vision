from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization


# Data load
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

num_classes = 10

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:], name='conv1'))
model.add(BatchNormalization(axis=3, name='bn_conv1'))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:], name='conv2'))
model.add(BatchNormalization(axis=3, name='bn_conv2'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', name='conv3'))
model.add(BatchNormalization(axis=3, name='bn_conv3'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same', name='conv4'))
model.add(BatchNormalization(axis=3, name='bn_conv4'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same', name='conv5'))
model.add(BatchNormalization(axis=3, name='bn_conv5'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same', name='conv6'))
model.add(BatchNormalization(axis=3, name='bn_conv6'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same', name='conv7'))
model.add(BatchNormalization(axis=3, name='bn_conv7'))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same', name='conv8'))
model.add(BatchNormalization(axis=3, name='bn_conv8'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, name='fc1'))
model.add(BatchNormalization(axis=1, name='bn_fc1'))
model.add(Activation('relu'))
model.add(Dense(num_classes, name='output'))
model.add(BatchNormalization(axis=1, name='bn_outptut'))
model.add(Activation('softmax'))

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Use ImageDataGenerator() for data augmentation
data_aug = ImageDataGenerator(
    rotation_range=15,  # randomly rotate images in the range
    width_shift_range=0.1,  # randomly shift images horizontally
    height_shift_range=0.1,  # randomly shift images vertically
    horizontal_flip=True,  # randomly flip images horizontally
    vertical_flip=False)  # randomly flip images vertically

from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

batch_size = 32
epochs = 30

# Use Adam optimizer
opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# Model compile (loss function = Categorical cross entropy)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# Save the model
filepath = 'model.{epoch:02d}-{val_loss:.5f}.hdf5'
model_chk = ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
                save_best_only=True, save_weights_only=True, mode='auto', period=1)

csv_log = CSVLogger('training.log')

# Train the model
history = model.fit_generator(data_aug.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    callbacks=[model_chk, csv_log])

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()