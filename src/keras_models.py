# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 21:30:31 2018

@authors: Barbiero Pietro and Ciravegna Gabriele
"""


"""Train a convnet on the MNIST database with ResNets.
ResNets are a bit overkill for this problem, but this illustrates how to use
the Residual wrapper on ConvNets.
See: https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
"""

from numpy import random
random.seed(42)  # @UndefinedVariable

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from resnet import Residual

import numpy as np

def keras_cnn_model(X_train, y_train, X_test, y_test, \
								epochs=50, batch_size=10, verbose=0):
	
	nb_classes = len( np.unique(y_train) )
	
	img_rows, img_cols = int( np.sqrt(X_train.shape[1]) ), int( np.sqrt(X_train.shape[1]) )
	pool_size = (2, 2)
	kernel_size = (3, 3)
	
#	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	
	if K.image_dim_ordering() == 'th':
	    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
	    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
	    input_shape = (1, img_rows, img_cols)
	else:
	    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
	    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
	    input_shape = (img_rows, img_cols, 1)
	
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255
#	print('X_train shape:', X_train.shape)
#	print(X_train.shape[0], 'train samples')
#	print(X_test.shape[0], 'test samples')
	
	# convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)
	
	# Model
	input_var = Input(shape=input_shape)
	
	conv1 = Conv2D(64, kernel_size, padding='same', activation='relu')(input_var)
	conv1 = MaxPooling2D(pool_size=pool_size)(conv1)
	conv1 = Conv2D(32, kernel_size, padding='same', activation='relu')(conv1)
	conv1 = MaxPooling2D(pool_size=pool_size)(conv1)
	conv1 = Conv2D(8, kernel_size, padding='same', activation='relu')(conv1)
	conv1 = MaxPooling2D(pool_size=pool_size)(conv1)
	
#	resnet = conv1
#	for _ in range(5):
#	    resnet = Residual(Convolution2D(8, kernel_size[0], kernel_size[1],
#	                                  border_mode='same'))(resnet)
#	    resnet = Activation('relu')(resnet)
#	mxpool = MaxPooling2D(pool_size=pool_size)(resnet)
	
	flat = Flatten()(conv1)
	dense = Dropout(0.5)(flat)
#	softmax = Dense(nb_classes, activation='relu')(dense)
#	dense = Dropout(0.5)(dense)
	softmax = Dense(nb_classes, activation='softmax')(dense)
	
	model = Model(inputs=[input_var], outputs=[softmax])
	model.compile(loss='categorical_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])
	
	model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
	          verbose=0, validation_data=(X_test, Y_test))
	#model.save('mnist_model.h5')
	loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
#	print('Test loss:', loss)
#	print('Test accuracy:', accuracy)
	
	return model, accuracy