from PIL import Image
import os
import numpy as np
import random

from keras.models import Sequential
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
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


def scale(im, SCALE_SIZE=255):
    if im.size[0] == im.size[1]:
        min_dim_i, max_dim_i = 0, 1
    else:
        min_dim_i = np.argmin(im.size)
        max_dim_i = np.argmax(im.size)

    scale_ratio = float(SCALE_SIZE) / float(im.size[min_dim_i])

    if scale_ratio >= 1 or scale_ratio <= 0:
        raise Exception("something terrible has happended")

    scaled_size = [0, 0]
    scaled_size[min_dim_i] = int(im.size[min_dim_i] * scale_ratio)
    scaled_size[max_dim_i] = int(im.size[max_dim_i] * scale_ratio)

    return im.resize(scaled_size, Image.ANTIALIAS)


def process_image(path, CROP_SIZE=224):
    im = Image.open(path)

    im = scale(im)

    w, h = im.size

    box = ((w - CROP_SIZE) / 2,
           (h - CROP_SIZE) / 2,
           (w + CROP_SIZE) / 2,
           (h + CROP_SIZE) / 2)

    im = im.crop(box)
    im = np.array(im).transpose(2, 1, 0)[0]
    return im / 255.


def alexnet(images_train, labels_train, images_test, labels_test):
    #AlexNet with batch normalization in Keras 
    #input image is 224x224

    model = Sequential()
    model.add(ZeroPadding2D((1,1), input_shape=(1, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(192, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    
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


def nnet_model(images_train, labels_train, images_test, labels_test):
    model = Sequential()

    # Then add a first layer 
    model.add(Dense(3000,input_shape=(30000,)))
    # Define the activation function to use on the nodes of that first layer
    model.add(Activation('relu'))

    # Second hidden layer
    model.add(Dense(1000))
    model.add(Activation('relu'))

    #Third hidden layer
    model.add(Dense(500))
    model.add(Activation('relu'))

    #Fourth hidden layer
    model.add(Dense(300))
    model.add(Activation('relu'))

    # Output layer with 10 categories (+using softmax)
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    adagrad = Adagrad(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=adagrad)

    model.fit(images_train,labels_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              show_accuracy=True,
              verbose=2,
              validation_data = (images_test, labels_test))

    score = model.evaluate(images_test,labels_test,show_accuracy=True,verbose=0)
    return model


def vgg_model(images_train, labels_train, images_test, labels_test):
    batch_size = 100
    nb_epoch = 10
    nb_classes = 2
    # Create the model
    vgg_model = Sequential()

    # On the very first layer, you must specify the input shape
    vgg_model.add(ZeroPadding2D((1,1),input_shape=(1,224,224))) 

    # Your first convolutional layer will have 64 3x3 filters,
    # and will use a relu activation function
    vgg_model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))

    # Once again you must add padding
    vgg_model.add(ZeroPadding2D((1,1)))
    vgg_model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))

    # Add a pooling layer with window size 2x2
    # The stride indicates the distance between each pooled window
    vgg_model.add(MaxPooling2D((2,2), strides=(2,2)))

    # Lots more Convolutional and Pooling layers

    vgg_model.add(ZeroPadding2D((1,1)))
    vgg_model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1')) 
    vgg_model.add(ZeroPadding2D((1,1)))
    vgg_model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    vgg_model.add(MaxPooling2D((2,2), strides=(2,2)))

    vgg_model.add(ZeroPadding2D((1,1)))
    vgg_model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1')) 
    vgg_model.add(ZeroPadding2D((1,1)))
    vgg_model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    vgg_model.add(ZeroPadding2D((1,1)))
    vgg_model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    vgg_model.add(MaxPooling2D((2,2), strides=(2,2)))

    # Flatten the input
    vgg_model.add(Flatten()) 

    # Add a fully connected layer with 4096 neurons
    vgg_model.add(Dense(4096, activation='relu')) 

    #Add a dropout layer
    vgg_model.add(Dropout(0.5))

    vgg_model.add(Dense(4096, activation='relu'))
    vgg_model.add(Dropout(0.5))

    vgg_model.add(Dense(2, activation='softmax'))

    #Compile the network we will explain this later
    sgd = SGD()
    vgg_model.compile(optimizer=sgd, loss='categorical_crossentropy')


    vgg_model.fit(images_train,labels_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  show_accuracy=True,
                  verbose=2,
                  validation_data = (images_test,labels_test)) 

    score = vgg_model.evaluate(images_test,labels_test,show_accuracy=True,verbose=0)
    
    return score


def _load_image(image_path):
    im = Image.open(image_path)
    im = im.resize((200,150), Image.ANTIALIAS)
    im_arr = np.asarray(im)
    im_tmp = im_arr.transpose(2, 1, 0)[0]
    return im_tmp.flatten() - 150.


def get_images_and_tags(dir, tag):
    files = os.listdir(PATH + dir)
    images, labels = [], []
    for file in files:
	image_arr = process_image(PATH + dir + file)
        #print image_arr.shape
	try:
	    images.append([image_arr])
	    labels.append(tag)
	except AttributeError:
	    continue

    images = np.array(images)
    labels = np.hstack(labels)

    return images, labels


label_map = {
    'DA': 0,
    'non-DA': 1
}

da_images, da_labels = get_images_and_tags('DA_images/', label_map['DA'])
print da_images.shape, da_labels.shape

non_da_images, non_da_labels = get_images_and_tags('non_DA_images/', label_map['non-DA'])
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

json_string = model.to_json()
open('my_model_v0_architecture.json', 'w').write(json_string)
model.save_weights('my_model_v0_weights.h5')
#print 'test accuracy: ', score[1]
