# This script has been designed to perform neural learning of archetypes
# by Alberto Tonda, Pietro Barbiero, and Gabriele Ciravegna, 2019 <alberto.tonda@gmail.com> <pietro.barbiero@studenti.polito.it>

#basic libraries

# tensorflow library

# sklearn library

# pandas

# basic libraries
import argparse
import copy
import datetime
import logging
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
# tensorflow library
import tensorflow as tf
from ghcore import Ghcore
# pandas
from pandas import read_csv
# sklearn library
from sklearn import datasets
from sklearn.datasets.samples_generator import make_blobs, make_circles, make_moons
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


import seaborn as sns
from matplotlib.colors import ListedColormap

warnings.filterwarnings("ignore")

def main(selectedDataset = "digits", max_het = 10, min_size = 5):

	# a few hard-coded values
	seed = 42

	# a list of classifiers
	allClassifiers = [
#			[SVC, "SVC", 1],
			[RandomForestClassifier, "RandomForestClassifier", 1],
##			[AdaBoostClassifier, "AdaBoostClassifier", 1],
			[BaggingClassifier, "BaggingClassifier", 1],
##			[ExtraTreesClassifier, "ExtraTreesClassifier", 1],
##			[GradientBoostingClassifier, "GradientBoostingClassifier", 1],
##			[SGDClassifier, "SGDClassifier", 1],
##			[PassiveAggressiveClassifier, "PassiveAggressiveClassifier", 1],
			[LogisticRegression, "LogisticRegression", 1],
			[RidgeClassifier, "RidgeClassifier", 1],
##			[LogisticRegressionCV, "LogisticRegressionCV", 1],
##			[RidgeClassifierCV, "RidgeClassifierCV", 0],
			]

	selectedClassifiers = [classifier[1] for classifier in allClassifiers]

	folder_name = "neural-archetypes-" + selectedDataset + "-" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
	if not os.path.exists(folder_name) :
		os.makedirs(folder_name)
	else :
		sys.stderr.write("Error: folder \"" + folder_name + "\" already exists. Aborting...\n")
		sys.exit(0)
	# open the logging file
	logfilename = os.path.join(folder_name, 'logfile.log')
	logger = setup_logger('logfile_' + folder_name, logfilename)
	logger.info("All results will be saved in folder \"%s\"" % folder_name)

	# load different datasets, prepare them for use
	logger.info("Preparing data...")
	# synthetic databases
	centers = [[1, 1], [-1, -1], [1, -1]]
	blobs_X, blobs_y = make_blobs(n_samples=400, centers=centers, n_features=2, cluster_std=0.6, random_state=seed)
	circles_X, circles_y = make_circles(n_samples=400, noise=0.15, factor=0.4, random_state=seed)
	moons_X, moons_y = make_moons(n_samples=400, noise=0.2, random_state=seed)
	iris = datasets.load_iris()
	digits = datasets.load_digits()
#	forest_X, forest_y = loadForestCoverageType() # local function
#	mnist_X, mnist_y = loadMNIST() # local function
	db = datasets.fetch_openml(name=selectedDataset, cache=False)

	dataList = [
			[blobs_X, blobs_y, 0, "blobs"],
			[circles_X, circles_y, 0, "circles"],
			[moons_X, moons_y, 0, "moons"],
	        [iris.data, iris.target, 0, "iris"],
	        [iris.data[:, 2:4], iris.target, 0, "iris2"],
	        [digits.data, digits.target, 0, "digits"],
#			[forest_X, forest_y, 0, "covtype"],
#			[mnist_X, mnist_y, 0, "mnist"]
			[db.data, db.target, 0, selectedDataset],
		      ]

	# argparse; all arguments are optional
	parser = argparse.ArgumentParser()

	parser.add_argument("--classifiers", "-c", nargs='+', help="Classifier(s) to be tested. Default: %s. Accepted values: %s" % (selectedClassifiers[0], [x[1] for x in allClassifiers]))
	parser.add_argument("--dataset", "-d", help="Dataset to be tested. Default: %s. Accepted values: %s" % (selectedDataset,[x[3] for x in dataList]))

	# finally, parse the arguments
	args = parser.parse_args()

	# a few checks on the (optional) inputs
	if args.dataset :
		selectedDataset = args.dataset
		if selectedDataset not in [x[3] for x in dataList] :
			logger.info("Error: dataset \"%s\" is not an accepted value. Accepted values: %s" % (selectedDataset, [x[3] for x in dataList]))
			sys.exit(0)

	if args.classifiers != None and len(args.classifiers) > 0 :
		selectedClassifiers = args.classifiers
		for c in selectedClassifiers :
			if c not in [x[1] for x in allClassifiers] :
				logger.info("Error: classifier \"%s\" is not an accepted value. Accepted values: %s" % (c, [x[1] for x in allClassifiers]))
				sys.exit(0)


	# print out the current settings
	logger.info("Settings of the experiment...")
	logger.info("Fixed random seed: %d" %(seed))
	logger.info("Selected dataset: %s; Selected classifier(s): %s" % (selectedDataset, selectedClassifiers))

	# create the list of classifiers
	classifierList = [ x for x in allClassifiers if x[1] in selectedClassifiers ]

	# pick the dataset
	db_index = -1
	for i in range(0, len(dataList)) :
		if dataList[i][3] == selectedDataset :
			db_index = i

	dbname = dataList[db_index][3]

	X, y = dataList[db_index][0], dataList[db_index][1]
	si = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
	X = si.fit_transform(X)
	le = LabelEncoder()
	y = le.fit_transform(y)
	y[y==2] = 1
	
	number_classes = np.unique(y).shape[0]

	logger.info("Creating train/test split...")
	from sklearn.model_selection import StratifiedKFold
	skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
	listOfSplits = [split for split in skf.split(X, y)]
	trainval_index, test_index = listOfSplits[0]
	X_trainval, X_test = X[trainval_index], X[test_index]
	y_trainval, y_test = y[trainval_index], y[test_index]
	logger.info("Training set: %d lines (%.2f%%); test set: %d lines (%.2f%%)" % (X_trainval.shape[0], (100.0 * float(X_trainval.shape[0]/X.shape[0])), X_test.shape[0], (100.0 * float(X_test.shape[0]/X.shape[0]))))
	skf2 = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
	list_of_splits = [split for split in skf2.split(X_trainval, y_trainval)]
	train_index, val_index = list_of_splits[0]
	X_train, X_val = X_trainval[train_index], X_trainval[val_index]
	y_train, y_val = y_trainval[train_index], y_trainval[val_index]

	# rescale data
	scaler = StandardScaler()
	sc = scaler.fit(X_train)
	X_trainval = sc.transform(X_trainval)
	X_train = sc.transform(X_train)
	X_val = sc.transform(X_val)
	X_test = sc.transform(X_test)

	for classifier in classifierList:

		classifier_name = classifier[1]

		# start creating folder name
		experiment_name = os.path.join(folder_name, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + "-neural-archetypes-" + dbname + "-" + classifier_name)
		if not os.path.exists(experiment_name) : os.makedirs(experiment_name)

		logger.info("Classifier used: " + classifier_name)
		modelB = copy.deepcopy(classifier[0](random_state=42))
		modelB.fit(X_trainval, y_trainval)
		trainAccuracyB = modelB.score(X_trainval, y_trainval)
		testAccuracyB = modelB.score(X_test, y_test)

		start = time.time()
		root, leaves, model, model2, len2, accuracy_train, X_core, y_core, outliers, pruned_nodes = Ghcore(X_train, y_train, X_val, y_val, X_test, y_test,
																							 max_height=2, min_epochs=10,
																							 max_heterogenity=max_het, heterogenity_decrease=0.05, epsilon_w=0.5,
																							 epsilon_n=0.05, min_size=min_size, min_accuracy=max(trainAccuracyB, 0.92), age_max=10,
																							 folder_name=experiment_name, classifier=classifier)
		end = time.time()
		exec_time = end - start

		leafFeats = [[leaf.y.size, leaf.heterogenity, np.mean(leaf.y)] for leaf in leaves]

		# accuracy baseline
		trainAccuracy = model.score(X_trainval, y_trainval)
		testAccuracy = model.score(X_test, y_test)
		trainAccuracy2 = model2.score(X_trainval, y_trainval)
		testAccuracy2 = model2.score(X_test, y_test)

		# select "best" individuals
		logger.info("Compute performances!")
		logger.info("Elapsed time (seconds): %.4f" %(exec_time))
		logger.info("Initial performance: train=%.4f, test=%.4f, size: %d" % (trainAccuracyB, testAccuracyB, X_train.shape[0]))
		logger.info("Final performance: train=%.4f, test=%.4f, size: %d" % (trainAccuracy, testAccuracy, X_core.shape[0]))
		logger.info("Final performace2: train=%.4f, test=%.4f, size: %d" % (trainAccuracy2, testAccuracy2, len2))

		if dbname == "mnist" or dbname == "digits":

			if dbname == "mnist":
				H, W = 28, 28
			if dbname == "digits":
				H, W = 8, 8

			logger.info("Now saving figures...")

			X_err, accuracy, model, fail_points, y_pred = evaluate_core(X_core, y_core, X_test, y_test, classifier[0], cname=classifier_name, SEED=seed)
			X_err_train, accuracy_train, model_train, fail_points_train, y_pred_train = evaluate_core(X_core, y_core, X_trainval, y_trainval, classifier[0], cname=classifier_name, SEED=seed)

			# save archetypes
			for index in range(0, len(y_core)):
				image = np.reshape(X_core[index, :], (H, W))
				plt.figure()
				plt.axis('off')
				plt.imshow(image, cmap=plt.cm.gray_r)
				plt.title('Label: %d' %(y_core[index]))
				plt.savefig( os.path.join(experiment_name, "digit_%d_idx_%d.pdf" %(y_core[index], index)) )
				plt.savefig( os.path.join(experiment_name, "digit_%d_idx_%d.png" %(y_core[index], index)) )
				plt.close()

			# save test errors
			e = 1
			for index in range(0, len(y_test)):
				if fail_points[index] == True:
					image = np.reshape(X_test[index, :], (H, W))
					plt.figure()
					plt.axis('off')
					plt.imshow(image, cmap=plt.cm.gray_r)
					plt.title('Label: %d - Prediction: %d' %(y_test[index], y_pred[index]))
					plt.savefig( os.path.join(experiment_name, "err_lab_%d_pred_%d_idx_%d.pdf" %(y_test[index], y_pred[index], e)) )
					plt.savefig( os.path.join(experiment_name, "err_lab_%d_pred_%d_idx_%d.png" %(y_test[index], y_pred[index], e)) )
					plt.close()
					e = e + 1

		
		figure = plt.figure()
		sns.boxplot(data=X_train, boxprops=dict(alpha=.3))
		g = sns.swarmplot(data=X_core)
		g.set_xticklabels(db.feature_names, rotation=45)
		plt.xticks(fontsize=15)
		plt.title(classifier_name + " - archetypes", fontsize=15)
		plt.tight_layout()
		plt.savefig( os.path.join(str(folder_name), str(classifier_name) + "_boxplot.pdf" ) )
		plt.savefig( os.path.join(str(folder_name), str(classifier_name) + "_boxplot.png" ) )
		plt.show(figure)
		
		
		X_err, accuracy, model, fail_points, y_pred = evaluate_core(X_core, y_core, X_test, y_test, classifier[0], cname=classifier_name, SEED=seed)
		
		figure = plt.figure()
		sns.boxplot(data=X_test, boxprops=dict(alpha=.3))
		g = sns.swarmplot(data=X_err)
		g.set_xticklabels(db.feature_names, rotation=45)
		plt.xticks(fontsize=15)
		plt.title(classifier_name + " - test errors", fontsize=15)
		plt.tight_layout()
		plt.savefig( os.path.join(str(folder_name), str(classifier_name) + "_errors_boxplot.pdf" ) )
		plt.savefig( os.path.join(str(folder_name), str(classifier_name) + "_errors_boxplot.png" ) )
		plt.show(figure)

		# plot decision boundaries if we have only 2 dimensions!
		if X.shape[1] == 2:

			X_err, accuracy, model, fail_points, y_pred = evaluate_core(X_core, y_core, X_test, y_test, classifier[0], cname=classifier_name, SEED=seed)
			X_err_train, accuracy_train, model_train, fail_points_train, y_pred_train = evaluate_core(X_core, y_core, X_train, y_train, classifier[0], cname=classifier_name, SEED=seed)

#			cmap = ListedColormap(sns.color_palette("bright", 3).as_hex())
#			xx, yy = make_meshgrid(X_train[:, 0], X_train[:, 1])
#			figure = plt.figure()
#			_, Z_0 = plot_contours(model, xx, yy, cmap=cmap, alpha=0.1)
#			plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, marker='.', alpha=0.3, label="training set")
#			plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap, marker='+', alpha=0.4, label="test set")
#			plt.scatter(X_core[:, 0], X_core[:, 1], c=y_core.squeeze(), cmap=cmap, marker='o', edgecolors='k', alpha=1, label="archetypes")
#			plt.scatter(X_err[:, 0], X_err[:, 1], marker='x', facecolors='k', edgecolors='k', alpha=1, label="test errors")
#			plt.legend()
#			plt.title("Decision buondaries (acc. %.4f) - %s" %(accuracy, classifier_name))
#			plt.savefig( os.path.join(experiment_name, "decision_boundaries.pdf") )
#			plt.savefig( os.path.join(experiment_name, "decision_boundaries.png") )
#			plt.close(figure)


			# using all samples in the training set
			X_core, y_core = X_trainval, y_trainval
			X_err, accuracy, model, fail_points, y_pred = evaluate_core(X_core, y_core, X_test, y_test, classifier[0], cname=classifier_name, SEED=seed)
			X_err_train, accuracy_train, model_train, fail_points_train, y_pred_train = evaluate_core(X_core, y_core, X_train, y_train, classifier[0], cname=classifier_name, SEED=seed)

			cmap = ListedColormap(sns.color_palette("bright", 3).as_hex())
			xx, yy = make_meshgrid(X_train[:, 0], X_train[:, 1])
			figure = plt.figure()
			_, Z_0 = plot_contours(model, xx, yy, cmap=cmap, alpha=0.2)
			plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, marker='.', alpha=0.3, label="training set")
			plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap, marker='+', alpha=0.4, label="test set")
			plt.scatter(X_err[:, 0], X_err[:, 1], marker='x', facecolors='k', edgecolors='k', alpha=1, label="test errors")
			plt.legend(fontsize=15)
			plt.title("%s\ntest acc. %.4f (full training)" %(classifier_name, accuracy), fontsize=15)
			plt.tight_layout()
			plt.savefig( os.path.join(experiment_name, str(classifier_name) + "_db_alltrain.pdf") )
			plt.savefig( os.path.join(experiment_name, str(classifier_name) + "_db_alltrain.png") )
			plt.show(figure)

	logger.handlers.pop()

	return


def setup_logger(name, log_file, level=logging.INFO):
	"""Function setup as many loggers as you want"""

	formatter = logging.Formatter('%(asctime)s %(message)s')
	handler = logging.FileHandler(log_file)
	handler.setFormatter(formatter)

	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.addHandler(handler)

	return logger

# utility function to load the covtype dataset
def loadForestCoverageType() :

	inputFile = "../data/covtype.csv"
	#logger.info("Loading file \"" + inputFile + "\"...")
	df_covtype = read_csv(inputFile, delimiter=',', header=None)

	# class is the last column
	covtype = df_covtype.as_matrix()
	X = covtype[:,:-1]
	y = covtype[:,-1].ravel()-1

	return X, y

def loadMNIST():
	mnist = tf.keras.datasets.mnist
	(x_train, y_train),(x_test, y_test) = mnist.load_data()

	X = np.concatenate((x_train, x_test))
	X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[1]))
	y = np.concatenate((y_train, y_test))

	return X, y

def make_meshgrid(x, y, h=.02):
	x_min, x_max = x.min() - 1, x.max() + 1
	y_min, y_max = y.min() - 1, y.max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
						np.arange(y_min, y_max, h))
	return xx, yy

def plot_contours(clf, xx, yy, **params):
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	out = plt.contourf(xx, yy, Z, **params)
	return out, Z

def evaluate_core(X_core, y_core, X, y, classifier, cname=None, SEED=0):

	if cname == "SVC":
		referenceClassifier = copy.deepcopy(classifier(random_state=SEED, probability=True))
	else:
		referenceClassifier = copy.deepcopy(classifier(random_state=SEED))
	referenceClassifier.fit(X_core, y_core)
	y_pred = referenceClassifier.predict(X)

	fail_points = y != y_pred

	X_err = X[fail_points]
	accuracy = accuracy_score( y, y_pred)

	return X_err, accuracy, referenceClassifier, fail_points, y_pred

if __name__ == "__main__" :
	dataList = [
#		["blobs", 10, 5],
		#["circles", 10, 5],
#		#["moons", 8, 8],
		#["iris", 5, 5],
		#["iris2", 3, 5],
#		["digits", 150, 5]
		#["covtype", 10, 5]
		#["mnist", 10, 5]
#		["credit-g", 10, 5]
		["cars", 10, 5]
#		["banknote-authentication", 10, 5]
		]
	for dataset in dataList:
		main(dataset[0], dataset[1], dataset[2])
	sys.exit()
