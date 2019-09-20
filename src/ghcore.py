# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 21:30:31 2018

@authors: Barbiero Pietro and Ciravegna Gabriele
"""


# sklearn library
from sklearn import datasets
from sklearn import decomposition
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
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

from keras.applications.resnet50 import ResNet50
from keras.models import load_model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Activation
from keras.callbacks import EarlyStopping

#from keras_models import keras_cnn_model

from anytree import NodeMixin, RenderTree, LevelOrderIter, search
import networkx as nx

import sys
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy import stats
from scipy.spatial.distance import pdist, cdist, euclidean
import csv
import time


import seaborn as sns
from matplotlib.colors import ListedColormap

#from master.src.main_test import folderName


def retrieve_n_class_color_cubic(N):
	'''
	retrive color code for N given classes
	Input: class number
	Output: list of RGB color code
	'''

	# manualy encode the top 8 colors
	# the order is intuitive to be used
	color_list = [
		(1, 0, 0),
		(0, 1, 0),
		(0, 0, 1),
		(1, 1, 0),
		(0, 1, 1),
		(1, 0, 1),
		(0, 0, 0),
		(1, 1, 1)
	]

	# if N is larger than 8 iteratively generate more random colors
	np.random.seed(1)  # pre-define the seed for consistency

	interval = 0.5
	while len(color_list) < N:
		the_list = []
		iterator = np.arange(0, 1.0001, interval)
		for i in iterator:
			for j in iterator:
				for k in iterator:
					if (i, j, k) not in color_list:
						the_list.append((i, j, k))
		the_list = list(set(the_list))
		np.random.shuffle(the_list)
		color_list.extend(the_list)
		interval = interval / 2.0

	return color_list[:N]

options = {
	'node_color': 'black',
	'node_size': 100,
	'width': 3,
}

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

	fail_points = y.squeeze() != y_pred

	X_err = X[fail_points]
	accuracy = accuracy_score(y, y_pred)

	return X_err, accuracy, referenceClassifier, fail_points, y_pred

class GhcoreNode(NodeMixin):
	
	def __init__(self, node_id, child_id=None, parent=None, X=[], y=[], sample_indeces=[], child_indeces=[], w=None, T=None, etime=1, graph=nx.Graph()):
		super().__init__()
		self.node_id = node_id
		self.child_id = child_id
		self.parent = parent
		self.X = X
		self.y = y
		self.sample_indeces = sample_indeces
		self.child_indeces = child_indeces
		self.w = w
		self.T = T
		self.etime = etime
		self.graph = graph
		self.heterogenity = 0
		if parent is None and len(y) > 0:
			self.heterogenity = np.sum(np.abs(self.X - self.w))
		
	def display(self):
		for pre, _, node in RenderTree(self):
			treestr = u"%s%s" % (pre, node.node_id)
			heterogenity = 0
			if len(node.y) > 0: 
				heterogenity = np.max( np.unique(node.y, return_counts=True) ) / len(node.y)
			print(treestr.ljust(8), 'n_samples: %d; heterogenity = %.2f' %(len(node.y), heterogenity), end='')
			print('')
	
	def set_w(self, w):
		self.w = w
		
	def set_T(self, T):
		self.T = T

	def set_up_child_indeces(self):
		self.child_indeces = [-1 for i in range(0, len(self.y))]
		
	def increment_elapsed_time(self):
		self.etime = self.etime + 1
	
	def update_child_id(self, child_node):
		child_node.child_id = len(self.children)-1
	
	def add_sample(self, sample, target, j, child):
		
		assert sample.shape[0] == 1
		if self != child.parent:
			child.X = np.concatenate((child.X, sample))
			child.y = np.concatenate((child.y, target))
			self.child_indeces[j] = -2
			child.sample_indeces.append(-1)
			child.child_indeces.append(-1)
			return 0

		old_child_idx = self.child_indeces[j]
		
#		print(child.child_id)
#		print(child.sample_indeces)
#		print(self.child_indeces)
#		print(j)
		if old_child_idx == -2: return -1
		
		if old_child_idx != child.child_id:
			
			if old_child_idx != -1:
				
				old_child = self.children[old_child_idx]
				sample_j = old_child.sample_indeces.index(j)

#				print(old_child.X.shape)
				old_child.X = np.delete(old_child.X, sample_j, axis=0)
				old_child.y = np.delete(old_child.y, sample_j, axis=0)
				old_child.sample_indeces.remove(j)
#				print(old_child.X.shape)
			
			if len(child.X) == 0:
				child.X, child.y = sample, target
			else:
				child.X = np.concatenate((child.X, sample))
				child.y = np.concatenate((child.y, target))
			
			child.sample_indeces.append(j)
			self.child_indeces[j] = child.child_id
			
#			print(child.sample_indeces)
#			print(self.child_indeces)
#			print()
		return 0
			
	def soft_competitive_learning(self, epsilon, sample):
		Delta = epsilon * (sample - self.w) / self.etime
		self.w = self.w + Delta
		
#	def soft_competitive_learning_2(self, epsilon_w, epsilon_n, sample, winner_1):
#		Delta = epsilon_w * (sample - self.w) / self.etime
#		self.w = self.w + Delta
		
	def update_threshold(self, child_node):
		neighbours_id = self.get_graph_neighbours(child_node)
		neighbours = search.findall(self, filter_=lambda node: node.node_id in neighbours_id, maxlevel=2)
		neighbours_W = np.array([node.w.squeeze() for node in neighbours])
		
		distances = np.sum( (child_node.w - neighbours_W)**2 , 1)
		
		if len(distances) > 1: average_distance = np.mean(distances)
		else: average_distance = distances[0]
		
		max_distance = np.max((self.T, average_distance))
		
		if max_distance == None: 
			print(1)
			
		child_node.set_T(max_distance)
		
		return neighbours

		
	def add_graph_node(self, node_id):
		self.graph.add_node(node_id)
		
	def add_graph_edge(self, winner_1, winner_2):
		self.graph.add_edge(winner_1.node_id, winner_2.node_id, weight=0)

	def update_graph_edge(self, winner_1, neighbour, age_max):
		self.graph[winner_1.node_id][neighbour.node_id]['weight'] += 1
		if self.graph[winner_1.node_id][neighbour.node_id]['weight'] > age_max:
			self.graph.remove_edge(winner_1.node_id, neighbour.node_id)
	
	def get_graph_neighbours(self, node):
		return list(self.graph.adj[node.node_id])
	
	def draw_graph(self):
		plt.figure()
		nx.draw(self.graph, with_labels=True, font_weight='bold')
		plt.show()
		
	def is_to_remove(self, node):
		return len(self.graph[node.node_id]) == 0
		
	def delete_child(self, node):
		self.graph.remove_node(node.node_id)
		self.children = [child for child in self.children if child!=node]

	def pruning(self, node_to_prune):
		if len(node_to_prune)== 0: return []
		assert self.parent is None
		leaves = []
		weights = []
		outliers = []
		for node in LevelOrderIter(self):
			if node.is_leaf:
				leaves.append(node)
				weights.append(node.w)
		for node in node_to_prune:
			for j in range(len(node.y)):
				sample = node.X[j, :]
				sample = np.reshape(sample, (1, len(sample)))
				target = node.y[j]
				target = np.reshape(target, (1, len(target)))
				distances = np.dot(np.asarray(weights).squeeze(), sample.T)
				nearest = leaves[np.argmin(distances)]
				if nearest.parent == node.parent:
					if nearest.T > np.min(distances):
						nearest.parent.add_sample(sample, target, node.sample_indeces[j], nearest)
					else:
						np.append(outliers, node.X[j])
				else:
					node.parent.add_sample(sample, target, node.sample_indeces[j], nearest)
			node.parent.delete_child(node)
		return outliers
		
	def plot_local_quantization(self, accuracy, n_leaves):
		
		nclass = len(np.unique(self.root.y))
		colors = np.array(retrieve_n_class_color_cubic(N=nclass))
		cy = np.array([colors[i].squeeze() for i in self.root.y-1])
		
		W = np.array([child.w.squeeze() for child in self.children])
		
		plt.figure()
		plt.scatter(self.root.X[:, 0], self.root.X[:, 1], c=cy, marker='.', alpha=0.3, label='voronoi set')
		plt.scatter(W[:, 0], W[:, 1], c='k', marker='o', label='gexin')
		plt.title('Ghcore - h=%d - #C=%d - acc.=%.2f' %(self.root.height, n_leaves, accuracy))
		plt.legend()
		plt.show()
		
	def plot_quantization(self, X_arch_core, y_arch_core, accuracy, leaves, folder_name, classifier_name):
		
		nclass = len(np.unique(self.root.y))
		colors = np.array(retrieve_n_class_color_cubic(N=nclass))
		cy = np.array([colors[i].squeeze() for i in self.root.y])
				
		ccore = np.array([colors[i].squeeze() for i in y_arch_core])
		
		cmap = ListedColormap(sns.color_palette("bright", 3).as_hex())
		plt.figure()
		plt.scatter(self.root.X[:, 0], self.root.X[:, 1], c=cy, cmap=cmap, marker='.', alpha=0.2, label='training set')
		plt.scatter(X_arch_core[:, 0], X_arch_core[:, 1], c=ccore, cmap=cmap, marker='o', label='archetypes')
		plt.title('GH-ARCH - %s - h=%d - acc.=%.2f' %(classifier_name, self.root.height, accuracy))
		plt.legend()
		plt.savefig("%s/ghcore_h%d.png" %(folder_name, self.root.height))
		plt.draw()
		

def predict_by_core(root, X_test, y_test, classifier):
	
	arch_core = []
	for node in LevelOrderIter(root):
		if node.is_leaf:
			if len(node.y) > 0:
				arch_core.append(node)
			else:
				arch_core.append(node.parent)
	_, arch_core_idx = np.unique([node.node_id for node in arch_core], return_index=True)
	arch_core = [ arch_core[i] for i in arch_core_idx]
	
	X_arch_core = np.array([node.w.squeeze() for node in arch_core])
	
	# centroids = []
	# for c in np.unique(root.y):
	# 	centroids.append( np.mean(root.X[root.y.squeeze()==c, :], axis=0) )
	# centroids = np.array(centroids)
	#
	# distances = cdist(X_arch_core, centroids)
	# y_arch_core = np.argmin(distances, axis=1)
	y_arch_core = []
	for node in arch_core:
		classes, n_samples = np.unique(node.y, return_counts=True)
		y_arch_core.append(classes[np.argmax(n_samples)])
		
	if len( np.unique(y_arch_core) ) < len( np.unique(y_test) ):
		accuracy = 0
		model = None
	else:
		n_classes = np.unique(y_arch_core).size
		n_cluster = range(len(y_arch_core))
		max_n_cluster_per_class = np.max(np.unique(y_arch_core, return_counts=True)[1])
		for i in range(n_classes):
			i_class_idx = [k for k in n_cluster if y_arch_core[k] == i]
			ith_class_n_cluster = len(i_class_idx)
			# Add each reference vector of a class for a number of time equale to maximum number
			# of cluster representing single class
			for j in range(max_n_cluster_per_class - ith_class_n_cluster):
				X_arch_core = np.append(X_arch_core,[X_arch_core[i_class_idx[j%len(i_class_idx)]]], axis=0)
				y_arch_core = np.append(y_arch_core,[y_arch_core[i_class_idx[j%len(i_class_idx)]]], axis=0)
		if False:
			model, accuracy = keras_cnn_model(X_arch_core, y_arch_core, X_test, y_test)
		else:
			model = copy.deepcopy( classifier[0](random_state=42) )
			model.fit(X_arch_core, y_arch_core)
			accuracy = model.score(X_test, y_test)
			
	
	return accuracy, arch_core, model, X_arch_core, y_arch_core

def Ghcore(X_train, y_train, X_val, y_val, X_test, y_test, max_height, min_epochs, max_heterogenity, epsilon_w, epsilon_n, min_size, min_accuracy,
		   folder_name, heterogenity_decrease, age_max, classifier):
	
	y_train = np.reshape(y_train, (len(y_train), 1))
	y_val = np.reshape(y_val, (len(y_val), 1))
	
	X_trainval = np.concatenate((X_train, X_val))
	y_trainval = np.concatenate((y_train, y_val))
	
	centroid_X = np.mean(X_train, axis=0)
	centroid_X = np.reshape(centroid_X, (1, len(centroid_X)))
	
	n_nodes = 0
	root = GhcoreNode('Node_' + str(n_nodes), parent=None, X=X_train, y=y_train, sample_indeces=[], w=centroid_X, T=np.Inf)
	root.set_up_child_indeces()
	n_nodes = n_nodes + 1
	
	k = 1
	parent = root
	accuracy = 0
	outliers = []
	pruned_nodes = 0
	model = None
	model2 = None

	while k < max_height and accuracy < min_accuracy:
#		print("Vertical growth - height = %d" %(k))
		
		leaves = [node for node in LevelOrderIter(root) if node.is_leaf and len(node.y) > min_size and node.heterogenity > max_heterogenity]
		
		n_leaves = len(leaves)
		if n_leaves == 0:
			break
		
		for i in range(0, n_leaves):
			
			parent = leaves[i]
			counter = 0
			epoch = 0
			heterogenity_rate = 0
			noise = np.random.uniform(0, 0.0001, parent.w.shape)
			n = GhcoreNode('Node_' + str(n_nodes), parent=parent, X=[], y=[], sample_indeces=[], w=parent.w+noise)
			parent.update_child_id(n)
			parent.add_graph_node('Node_' + str(n_nodes))
			n_nodes = n_nodes + 1
			n = GhcoreNode('Node_' + str(n_nodes), parent=parent, X=[], y=[], sample_indeces=[], w=parent.w-noise)
			parent.update_child_id(n)
			parent.add_graph_node('Node_' + str(n_nodes))
			n_nodes = n_nodes + 1
			
			while epoch < min_epochs and heterogenity_rate < heterogenity_decrease:
				
				first_time = True
				
				# learning process
				for j in range(0, len(parent.y)):
						
#					if k > 2 and epoch > 0 and j > 3:
#						print(epoch)
#						print(j)
					
					sample = parent.X[j, :]
					sample = np.reshape(sample, (1, len(sample)))
					target = parent.y[j]
					target = np.reshape(target, (1, len(target)))
					
					W = np.array([leaf.w.squeeze() for leaf in parent.children])
					distances = np.sum( (sample - W)**2 , 1)
					winner_1_idx = np.argmin(distances)
					distance = np.sqrt(distances[winner_1_idx])
					distances[winner_1_idx] = np.inf
					winner_2_idx = np.argmin(distances)
					
					winner_1 = parent.children[winner_1_idx]
					winner_2 = parent.children[winner_2_idx]
					
					if first_time:
						first_time = False
						avgT = np.mean(pdist(parent.X))
						
						if epoch == 0:
							winner_1.set_T(avgT)
							winner_2.set_T(avgT)
							parent.set_T(avgT)
							
						if parent.add_sample(sample, target, j, winner_1) == -1: continue
						winner_1.increment_elapsed_time()
						winner_1.soft_competitive_learning(epsilon_w, sample)
						
						parent.add_graph_edge(winner_1, winner_2)
						
#						parent.draw_graph()
						
					else:
						
						if False: #parent.get_graph_neighbours(winner_1) >= parent.X.shape[1]:
							
							# use convex hull
							1
							
						else:
							if winner_1.T == None: 
								print(1)
							explainable = distance < winner_1.T
							
						if explainable:
							
							if parent.add_sample(sample, target, j, winner_1) == -1: continue
							winner_1.increment_elapsed_time()
							winner_1.soft_competitive_learning(epsilon_w, sample)
							
							parent.add_graph_edge(winner_1, winner_2)
							
#							parent.draw_graph()
							
							neighbours = parent.update_threshold(winner_1)
							
							for neighbour in neighbours:
								neighbour.soft_competitive_learning(epsilon_n, sample)
								parent.update_threshold(neighbour)
								if neighbour != winner_2:
									parent.update_graph_edge(winner_1, neighbour, age_max)
								
						else:
							
							new_node = GhcoreNode('Node_' + str(n_nodes), parent=parent, X=[], y=[], sample_indeces=[], w=sample)
							parent.update_child_id(new_node)
							parent.add_graph_node('Node_' + str(n_nodes))
							n_nodes = n_nodes + 1
							
							parent.add_sample(sample, target, j, new_node)
							
							new_node.set_T(parent.T)
							counter = 0
							
							if new_node.T == None: 
								print(1)

				if False:
					#node pruning
					nodes_to_prune = []
					for node in parent.children:
						if parent.is_to_remove(node):
							nodes_to_prune.append(node)
					if len(nodes_to_prune) > 0:
						outliers.append(root.pruning(nodes_to_prune))
						pruned_nodes += len(nodes_to_prune)
				heterogenities = []
				for node in parent.children:
					if len(node.y) > 0:
						node.heterogenity = np.sum(np.abs(node.X - node.w))
						heterogenities.append(node.heterogenity)
				avg_heterogenity = np.mean(heterogenities)
				heterogenity_rate = np.abs(avg_heterogenity-parent.heterogenity)/parent.heterogenity
				epoch = epoch + 1
				counter = counter + 1
			
			for child in parent.children:
				child.set_up_child_indeces()
#			parent.draw_graph()
		if not model is None:
			model2 = model
			len2 = len(y_arch_core)
		accuracy, leaves, model, X_arch_core, y_arch_core = predict_by_core(root, np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)), classifier)
		
		if X_test.shape[1] == 2:
#			parent.plot_quantization(X_arch_core, y_arch_core, accuracy, leaves, folder_name, classifier[1])
			
			X_err_test, accuracy_test, model_test, fail_points_test, y_pred_test = evaluate_core(X_arch_core, y_arch_core, X_test, y_test, classifier[0], cname=classifier[1], SEED=42)
			X_err_train, accuracy_train, model_train, fail_points_train, y_pred_train = evaluate_core(X_arch_core, y_arch_core, X_trainval, y_trainval, classifier[0], cname=classifier[1], SEED=42)
			
			cmap = ListedColormap(sns.color_palette("bright", 3).as_hex())
			xx, yy = make_meshgrid(X_train[:, 0], X_train[:, 1])
			figure = plt.figure()
			_, Z_0 = plot_contours(model, xx, yy, cmap=cmap, alpha=0.2)
			plt.scatter(X_trainval[:, 0], X_trainval[:, 1], c=y_trainval.squeeze(), cmap=cmap, marker='.', alpha=0.3, label="training set")
			plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap, marker='+', alpha=0.4, label="test test")
			plt.scatter(X_arch_core[:, 0], X_arch_core[:, 1], c=y_arch_core, cmap=cmap, marker='o', edgecolors='k', alpha=1, label="archetypes")
			plt.scatter(X_err_test[:, 0], X_err_test[:, 1], marker='x', facecolors='k', edgecolors='k', alpha=1, label="test errors")
			plt.legend(fontsize=15)
			plt.title("%s\nval.acc. %.4f - test acc. %.4f" %(classifier[1], accuracy, accuracy_test), fontsize=15)
			plt.tight_layout()
			plt.savefig( os.path.join(folder_name, classifier[1] + "_decision_boundaries" + str(k) + ".pdf") )
			plt.savefig( os.path.join(folder_name, classifier[1] + "_decision_boundaries" + str(k) + ".png") )
			plt.show(figure)
			
#		else:
		print('Ghcore: height=%d - coreset size=%d - accuracy=%.2f' %(root.height, len(y_arch_core), accuracy))
			
#		accuracy = predict_by_core(root, X_test, y_test)
				
		k = k + 1

#	root.display()
#	X_train = np.concatenate((X, X_test))
#	y = np.concatenate((y, y_test))
	accuracy, leaves, model, X_arch_core, y_arch_core = predict_by_core(root, X_trainval, y_trainval, classifier)
#	print('\nGhcore: height=%d - coreset size=%d - accuracy=%.2f\n' %(root.height, len(leaves), accuracy))
	leaves = [node for node in LevelOrderIter(root) if node.is_leaf and len(node.y) > 0]
	
	if model2 == None: 
		model2 = model
		len2 = len(y_arch_core)
	return root, leaves, model, model2, len2, accuracy, X_arch_core, y_arch_core, outliers, pruned_nodes


def main() :

	print("Loading datasets...")
	X, y = datasets.load_iris(return_X_y=True)
	X_train = X[:, 2:4]
	
	skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
	list_of_splits = [split for split in skf.split(X, y)]
	train_index, test_indeX_train = list_of_splits[0]
	X_train, y_train = X[train_index], y[train_index]
	X_test, y_test = X[test_index], y[test_index]

	folderName = "../results/" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + "-neuro-core"
	if not os.path.exists(folderName):
		os.makedirs(folderName)
	else:
		sys.stderr.write("Error: folder \"" + folderName + "\" already exists. Aborting...\n")
		sys.exit(0)

	nn = Ghcore(X_train, y_train, X_test, y_test, max_height=6, min_epochs=3, max_heterogenity=0.6, heterogenity_decrease = 0.25, epsilon_w=0.2, epsilon_n=0.01, min_size=5, min_accuracy=0.8, folder_name=folderName)
	
	return

if __name__ == "__main__" :
	sys.exit( main() )