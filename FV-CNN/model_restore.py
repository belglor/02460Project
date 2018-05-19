import pickle
import numpy as np
import matplotlib.pyplot as plt
import fvcnn_full as fvcnn 
import tensorflow as tf

#Create network
sess = tf.Session()
imgs = tf.placeholder(tf.float32, [None, None, None, 3])
network = fvcnn.fvcnn(imgs, 'vgg16_weights.npz', sess)
    
#Load/restore GMM weights
f = open("weights","rb")
weights = pickle.load(f)
network.weights = weights
f.close()

#Load/restore GMM means
f = open("means","rb")
means = pickle.load(f)
network.means = means
f.close()

#Load/restore GMM means
f = open("covs","rb")
covs = pickle.load(f)
network.covs = covs
f.close()

#Reload train_fv and retrain SVM
#Load/restore GMM means
f = open("train_fv","rb")
train_fv = pickle.load(f)
f.close()

#Load/restore GMM means
f = open("train_loaded_labels","rb")
train_loaded_labels = pickle.load(f)
f.close()

import sklearn.svm
print("-> Creating the classifier:")
clf = sklearn.svm.LinearSVC(penalty='l2', loss='hinge',
                            dual=True, tol=0.0001, C=1.0, multi_class='ovr',
                            fit_intercept=True, intercept_scaling=1,
                            class_weight=None, verbose=0,
                            random_state=None, max_iter=1000)
print("-> Training...")
clf.fit(train_fv,train_loaded_labels)
print("-> Completed!")

