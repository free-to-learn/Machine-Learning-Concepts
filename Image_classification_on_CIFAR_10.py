# Image classification on CIFAR 10

import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Dropout, MaxPooling2D, Conv2D, Convolution2D, BatchNormalization, Activation
import matplotlib.pyplot as plt
from keras.utils import to_categorical

import warnings
warnings.filterwarnings('ignore')

#loading data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#priting features shape
print('x_train shape:', len(x_train))
print('x_test shape:', x_test.shape)
print('x_test shape:', len(x_test))

#priting labels shape
print('y_train shape:', y_train.shape)
print('y_train shape:', len(y_train))
print('y_test shape:', y_test.shape)
print('y_test shape:', len(y_test))

#all classes images
for i in range(10):
    plt.imshow(x_train[i])
    plt.show()


#one hot encoding on y column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

y_train = y_train/255
y_test = y_test/255

#changing datatype fo columns
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

#building a model
classifier = Sequential()

#Convolution Layers with regularization
classifier.add(Convolution2D(32, (3,3), input_shape=(32, 32, 3)))
classifier.add(Activation('relu'))
classifier.add(BatchNormalization())
classifier.add(Convolution2D(32, (3,3), input_shape=(32, 32, 3)))
classifier.add(Activation('relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(3,3)))
classifier.add(Dropout(0.4))

classifier.add(Convolution2D(64, (3,3)))
classifier.add(Activation('relu'))
classifier.add(BatchNormalization())
classifier.add(Convolution2D(64, (3,3)))
classifier.add(Activation('relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(3,3)))
classifier.add(Dropout(0.5))

#Adding Dense Layers
classifier.add(Flatten())
classifier.add(Dense(units=1024))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=10))
classifier.add(Activation('softmax'))

#Compiling model
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

datagen.fit(x_train)

classifier.fit_generator(datagen.flow(x_train, y_train, batch_size=64),
                        epochs=128, validation_data=(x_test, y_test), steps_per_epoch=len(x_train) / 32)

#saving sn object
classifier.save('Object_classification_model.h5')

# Score trained model.
scores = classifier.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
