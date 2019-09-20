# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 17:02:42 2018

@author:  Barbiero Pietro and Ciravegna Gabriele
"""

# sklearn library
from sklearn import datasets
from sklearn import decomposition
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
# here are all the classifiers
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical

from ghcore import Ghcore

from anytree import NodeMixin, RenderTree, LevelOrderIter, search
import networkx as nx

import sys
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy import stats
from scipy.spatial.distance import pdist, euclidean
import csv
import time
import logging
#from keras_models import keras_cnn_model


def setup_logger(name, log_file, level=logging.INFO):
	"""Function setup as many loggers as you want"""
	
	formatter = logging.Formatter('%(asctime)s %(message)s')
	handler = logging.FileHandler(log_file)        
	handler.setFormatter(formatter)
	
	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.addHandler(handler)
	
	return logger

SEED = 42

folderName = "../results/" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + "-neuro-core"
if not os.path.exists(folderName) : 
	os.makedirs(folderName)
else :
	sys.stderr.write("Error: folder \"" + folderName + "\" already exists. Aborting...\n")
	sys.exit(0)

# open the logging file
logfilename = os.path.join(folderName, 'neuro_core.log')
logging = setup_logger('neuro_core_log', logfilename)



# Load data

logging.info("Loading datasets...")

data_list = []

## IRIS
#X, y = datasets.load_iris(return_X_y=True)
#data_list.append([X, y, "iris4", 4])
#
## IRIS2
#X = X[:, 2:4]
#data_list.append([X, y, "iris2", 4])
#
## DIGITS
X, y = datasets.load_digits(return_X_y=True)
data_list.append([X, y, "digits", 10])

# MNIST
#mnist_train = np.genfromtxt('../data/mnist_train.csv', delimiter=',')
#mnist_test = np.genfromtxt('../data/mnist_test.csv', delimiter=',')
#X_train = mnist_train[1:, 1:]
#y_train = mnist_train[1:, 0]
#X_test = mnist_test[1:, 1:]
#y_test = mnist_test[1:, 0]
#X, y = np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test))
#data_list.append([X, y, "mnist", 10])


data_index = 0


for dataset in data_list:
	
	X, y, db_name, N_SPLITS = dataset[0], dataset[1], dataset[2], dataset[3]
	
	accuracy_list = []
	base_accuracy_list = []
	size_list = []
	height_list = []
	
	logging.info("GhCore on '%s' database", db_name)
	logging.info("#samples = %d; #features = %d" %(X.shape[0], X.shape[1]))
			  
	
	logging.info("Creting train/test split...")
	skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
	for trainval_index, test_index in skf.split(X, y):
		X_trainval, X_test = X[trainval_index], X[test_index]
		y_trainval, y_test = y[trainval_index], y[test_index]
		
		# accuracy baseline
		if True:
			model = PassiveAggressiveClassifier()
			model.fit(X_trainval, y_trainval)
			base_accuracy_test = model.score(X_test, y_test)
		else:
			model, base_accuracy_test = keras_cnn_model(X_trainval, y_trainval, X_test, y_test)
		print("Baseline accuracy: %.4f" %(base_accuracy_test))
		base_accuracy_list.append(base_accuracy_test)
		
		# try with GhCore
		skf2 = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
		list_of_splits = [split for split in skf.split(X_trainval, y_trainval)]
		train_index, val_index = list_of_splits[0]
		X_train, X_val = X_trainval[train_index], X_trainval[val_index]
		y_train, y_val = y_trainval[train_index], y_trainval[val_index]
		
		root, _, model, accuracy_train, X_arch_core, y_arch_core, outliers, pruned_nodes = Ghcore(X_train, y_train, X_val, y_val, \
																 max_height=20, min_epochs=10, \
																 max_heterogenity=20000, heterogenity_decrease = 0.25, epsilon_w=0.5, \
																 epsilon_n=0.05, min_size=5, min_accuracy=0.9, age_max= 10, \
																 folder_name=folderName)
		if True:
			accuracy_test = model.score(X_test, y_test)
		else:
			img_rows, img_cols = int( np.sqrt(X_train.shape[1]) ), int( np.sqrt(X_train.shape[1]) )
			X_test_cnn = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
			X_test_cnn = X_test_cnn.astype('float32')
			X_test_cnn /= 255
			y_test_cnn = to_categorical(y_test, len( np.unique(y_train) ))
			accuracy_loss, accuracy_test = model.evaluate(X_test_cnn, y_test_cnn, verbose=0)
		
		logging.info("\ttraining score: %.4f ; test score: %.4f", accuracy_train, accuracy_test)
		print("training score: "+str(accuracy_train)+" ; test score:" + str(accuracy_test)+" ; #outliers:" + str(len(outliers))+" ; pruned nodes:" + str(pruned_nodes))
		accuracy_list.append(accuracy_test)
		size_list.append(len(y_arch_core))
		height_list.append(root.height)
	
	logging.info("Average coreset size: %.4f (+/- %.4f)" % (np.mean(size_list), np.std(size_list)))
	logging.info("Average GhCore height: %.4f (+/- %.4f)" % (np.mean(height_list), np.std(height_list)))
	logging.info("Average performance (test) of GhCore: %.4f (+/- %.4f)" % (np.mean(accuracy_list), np.std(accuracy_list)))
	logging.info("Average baseline performance (test): %.4f (+/- %.4f)" % (np.mean(base_accuracy_list), np.std(base_accuracy_list)))
	logging.handlers.pop()
	
#	if db_name == "mnist":
#		H, W = 28, 28
#		for index in range(0, len(y_arch_core)):
#			image = np.reshape(X_arch_core[index, :], (H, W))
#			plt.figure()
#			plt.axis('off')
#			plt.imshow(image, cmap=plt.cm.gray_r)
#			plt.title('Class: %d' %(y_arch_core[index]))
#			plt.savefig("%s/mnist_%d_%d.png" %(folderName, y_arch_core[index], index))
#			plt.show()
#			
#	if db_name == "digits":
#		H, W = 8, 8
#		for index in range(0, len(y_arch_core)):
#			image = np.reshape(X_arch_core[index, :], (H, W))
#			plt.figure()
#			plt.axis('off')
#			plt.imshow(image, cmap=plt.cm.gray_r)
#			plt.title('Class: %d' %(y_arch_core[index]))
#			plt.savefig("%s/digits_%d_%d.png" %(folderName, y_arch_core[index], index))
#			plt.show()
		
	
	




