from PIL import Image
import os
import numpy as np
import random

from keras.models import Sequential
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.convolutional import ZeroPadding1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adagrad, Adam
from keras.utils import np_utils

from sklearn.cross_validation import train_test_split

#import theano
#theano.config.device = 'gpu'
#theano.config.floatX = 'float32'

PATH = '/data/sandragreiss/astro/data/'
batch_size = 100
nb_epoch = 10
nb_classes = 2

LAMBDA_MIN = 3820
LAMBDA_MAX = 9000
NORM_MIN = 0
NORM_MAX = 1


def _get_data(directory, tag):
    files = os.listdir(PATH + directory)
    print len(files)
    total = 0
    all_fluxes, all_labels = [], []

    for fle in files:
        x = np.loadtxt(PATH + directory + fle, usecols=[0])
        y = np.loadtxt(PATH + directory + fle, usecols=[1])

        if x.min() < LAMBDA_MIN and x.max() > LAMBDA_MAX:
            total += 1
            x = x[(x >= LAMBDA_MIN) & (x <= LAMBDA_MAX)]
            y = y[(x >= LAMBDA_MIN) & (x <= LAMBDA_MAX)]

            A = y.min()
            B = y.max()

            normalized_y = (NORM_MIN + (y - A) * (NORM_MAX - NORM_MIN)) / (B - A)

            all_fluxes.append(normalized_y)
            all_labels.append(tag)

    print total

    all_fluxes = np.vstack(all_fluxes)
    all_labels = np.hstack(all_labels)

    return all_fluxes, all_labels

def alexnet(images_train, labels_train, images_test, labels_test):
    #AlexNet with batch normalization in Keras 

    model = Sequential()
    model.add(ZeroPadding1D((1), input_shape=(1, 3722)))
    model.add(Convolution1D(64, 3, activation='relu'))
    model.add(MaxPooling1D((2)))#, strides=(2,2)))
    #model.add(Activation('relu'))

    model.add(ZeroPadding1D((1)))
    model.add(Convolution1D(128, 3, activation='relu'))
    model.add(MaxPooling1D((2)))#, strides=(2,2)))
    #model.add(Convolution2D(128, 3, 3))
    #model.add(BatchNormalization((128,115,115)))
    #model.add(Activation('relu'))

    model.add(ZeroPadding1D((1)))
    model.add(Convolution1D(192, 3, activation='relu'))
    model.add(MaxPooling1D((2)))#, strides=(2,2)))
    #model.add(Convolution2D(192, 3, 3, border_mode='same'))
    #model.add(BatchNormalization((128,112,112)))
    #model.add(Activation('relu'))

    model.add(ZeroPadding1D((1)))
    model.add(Convolution1D(256, 3, activation='relu'))
    model.add(MaxPooling1D((2)))#, strides=(2,2)))
    #model.add(Convolution2D(256, 3, 3, border_mode='same'))
    #model.add(BatchNormalization((128,108,108)))
    #model.add(Activation('relu'))

    model.add(ZeroPadding1D((1)))
    model.add(Convolution1D(512, 3, activation='relu'))
    model.add(MaxPooling1D((2)))
    #model.add(MaxPooling2D(poolsize=(3,3)))

    #model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(4096, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(BatchNormalization(4096))
    #model.add(Activation('relu'))
    #model.add(Dense(4096, init='normal'))
    #model.add(BatchNormalization(4096))
    #model.add(Activation('relu'))
    model.add(Dense(2, activation='softmax'))
    #model.add(BatchNormalization(2))
    #model.add(Activation('softmax'))

    #adagrad = Adagrad(lr=0.01)
    #adam = Adam()
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    model.fit(images_train,labels_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              show_accuracy=True,
              verbose=2,
              validation_data = (images_test, labels_test))

    score = model.evaluate(images_test,labels_test,show_accuracy=True,verbose=0)

    return score, model


label_map = {
    'DA': 0,
    'non-DA': 1
}

da_images, da_labels = _get_data('DA/', label_map['DA'])
print da_images.shape, da_labels.shape

non_da_images, non_da_labels = _get_data('non-DA/', label_map['non-DA'])
print non_da_images.shape, non_da_labels.shape

all_images = np.concatenate((da_images, non_da_images[:8000]))
all_labels = np.concatenate((da_labels, non_da_labels[:8000]))
all_labels.astype('uint8')

print all_images.shape, all_labels.shape

def shuffle_data(N):
    s = []
    while (len(s) < N) :
	n = random.randint(0, N)
	if n not in s:
	    s.append(n)

    return s

def split_data(all_images, all_labels):
    x_train, y_train, x_test, y_test = [], [], [], []
    ind = shuffle_data(len(all_labels)-1)
    print len(ind), len(all_labels)
    for i in ind:
	train_or_test = 'test' if random.randint(1, 5) == 1 else 'train'
        if train_or_test == 'train':
	    x_train.append(all_images[i])
	    y_train.append(all_labels[i])
	else:
	    x_test.append(all_images[i])
	    y_test.append(all_labels[i])

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

	
images_train, labels_train, images_test, labels_test = split_data(all_images, all_labels)

labels_train = np_utils.to_categorical(labels_train, nb_classes)
labels_test  = np_utils.to_categorical(labels_test, nb_classes)

print images_train.shape, images_test.shape
print labels_train.shape, labels_test.shape

score, model = alexnet(images_train, labels_train, images_test, labels_test)
print score[1]

#json_string = model.to_json()
#open('my_model_v0_architecture.json', 'w').write(json_string)
#model.save_weights('my_model_v0_weights.h5')
#print 'test accuracy: ', score[1]
