########################################################################################
# VGG16 implementation in TensorFlow taken and adapted from http://www.cs.toronto.edu/~frossard/post/vgg16/
# Copyright Davi Frossard, 2016
#
# Fisher Vector implementation taken and adapted from https://github.com/jacobgil/pyfishervector
#
########################################################################################

import tensorflow as tf
import numpy as np
from scipy.misc import imread
import sys
import glob
from sklearn import mixture
from scipy.stats import multivariate_normal
from sklearn.model_selection import KFold
import os
from types import ModuleType

# FVCNN class builder
# When called, the class uses tf layers to build a VGG16 CNN network (note that only the
# convolutional layers are built). The class contains methods to compute the corresponding
# Fisher Vector from the output of the CNN layer

# TO DO:
#   - Separate training and predicting: need to differentiate between a training loop carried
#     out on a dataset and a simple forward pass and prediction of an image (SO GMM IS
#     TRAINED ONLY IN THE TRAINING LOOP, then it stays fixed for prediction)
#   - Implement SVM classification on the FVs
#   - Implement crisp region proposal for segmentation
#
# TO FIX:
#   - Is GMM carried out properly? Are things correctly concatenated?
#   - Is GMM working properly? Probability assignment looks weird (same probability assignment to all points?)
#   - Are FV working properly?
#
#




class fvcnn:
    # Class builder
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.convlayers()
        self.descripts = self.pool5;
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)
        #Have class variables to store the learned GMM dictionary
        self.means = []
        self.covs = []
        self.weights = []

    # Convolutional layer builder
    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    # Pre-trained VGG16 weights loader
    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        #Remove fully connected
        keys = keys[:-6]
        for i, k in enumerate(keys):
            print( i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))

    ### DATA LOADING METHODS ###

    def get_data(self, folder, extns='*.jpg'):
        files = glob.glob(folder+extns)
        return files

    def get_descriptors(self,feed_imgs):
        descripts = []
        for i in range(len(feed_imgs)):
            if(i%1 == 0):
                print("\t - Propagating picture #" +str(i)+ "/" + str(len(feed_imgs)))
                sys.stdout.flush()
            food = feed_imgs[i]
            food = np.reshape(food, [1, food.shape[0], food.shape[1], food.shape[2] ])
            descripts.append(sess.run(self.descripts, feed_dict={self.imgs: food}))
        return descripts

    def forward_pass(self, files, pics_to_load=None):
        #If unspecified, load the whole dataset
        loaded_paths = []
        feed_imgs = []
        if pics_to_load == None:
            pics_to_load = len(files)
        for i in range(pics_to_load):
            if(i%10 == 0):
                print("\t - Loading picture #" +str(i)+ "/" + str(pics_to_load))
                sys.stdout.flush()
            img = imread(files[i], mode ='RGB')
            feed_imgs.append(img)
            loaded_paths.append(files[i])
        print("\n")
        mat_descripts = self.get_descriptors(feed_imgs)
        #Return also stacked version of descripts
        stacked_descripts = []
        for x in mat_descripts:
            stacked_descripts.append(np.concatenate(np.concatenate(x)))
        stacked_descripts = np.vstack(stacked_descripts)
        return mat_descripts, stacked_descripts, loaded_paths

    # Fit a GMM to descriptors (HOW ARE DESCRIPTORS FED?)
    def generate_GMM(self, descriptors, N):
        gmm = mixture.GaussianMixture(N,covariance_type='diag')
        gmm.fit(descriptors)
        self.means = np.float32(gmm.means_)
        self.covs =  np.float32(gmm.covariances_)
        self.weights = np.float32(gmm.weights_)
        return

    # Compute moments of gaussian distribution
    def likelihood_moment(self, x, ytk, moment):
        	x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
        	return x_moment * ytk

    # Given a GMM model (given by means, covs and mixture weights) and samples, compute
    # the samples statistics wro the given GMM
    def likelihood_statistics(self, samples):
        means = self.means
        covs = self.covs
        weights = self.weights
        n_samples = samples.shape[0]
        dim_samples = samples.shape[1]
        n_mixtures = weights.shape[0]
        gaussians = np.zeros([n_samples, n_mixtures])
        #BE CAREFUL: I'M ALLOWING SINGULAR MATRICES FOR COVARIANCE
        # Create a ultivariate normal for each of the mixture components
        g = []
        for k in range(n_mixtures):
            g.append(multivariate_normal(mean=means[k], cov=covs[k], allow_singular = True))
        #For each observation, compute the likelihood of being in one of the mixture
        for index in range(n_samples):
            for k in range(n_mixtures):
                gaussians[index,k] = g[k].logpdf(samples[index,:])

        s0 = np.zeros([n_mixtures])
        s1 = np.zeros([n_mixtures,dim_samples])
        s2 = np.zeros([n_mixtures,dim_samples])
        probs = np.multiply(gaussians, weights.T)
        # For each observation, create probability matrix
        for index in range(n_samples):
            probs[index,:] = probs[index,:] / sum(probs[index,:])

        for index in range(n_samples):
            for k in range(n_mixtures):
                s0[k] = s0[k] + self.likelihood_moment(samples[index,:], probs[index,k], 0)
                s1[k] = s1[k,:] + self.likelihood_moment(samples[index,:], probs[index,k], 1)
                s2[k] = s2[k,:] + self.likelihood_moment(samples[index,:], probs[index,k], 2)

        return s0, s1, s2

    def fisher_vector_alphas(self, s0, s1, s2, means, covs, w, T):
        	return np.float32([((s0[k] - T * w[k]) / np.sqrt(w[k]) ) for k in range(len(w))])

    def fisher_vector_mus(self, s0, s1, s2, means, sigma, w, T):
        	return np.float32([(s1[k,:] - means[k] * s0[k]) / (np.sqrt(w[k] * sigma[k])) for k in range(0, len(w))])

    def fisher_vector_sigma(self, s0, s1, s2, means, sigma, w, T):
        	return np.float32([(s2[k,:] - 2 * means[k]*s1[k,:]  + (means[k]*means[k] - sigma[k]) * s0[k]) / (np.sqrt(2*w[k])*sigma[k])  for k in range(len(w))])

    def normalize(self, fisher_vector):
        	v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
        	return v / np.sqrt(np.dot(v, v))

    # Compute a sample's Fisher Vector
    def fisher_vector(self, samples):
        means = self.means
        covs = self.covs
        weights = self.weights
        s0, s1, s2 =  self.likelihood_statistics(samples)
        T = samples.shape[0]
        a = self.fisher_vector_alphas(s0, s1, s2, means, covs, weights, T)
        b = self.fisher_vector_mus(s0, s1, s2, means, covs, weights, T)
        c = self.fisher_vector_sigma(s0, s1, s2, means, covs, weights, T)
        fv = np.concatenate([a, np.concatenate(b), np.concatenate(c)])
        fv = self.normalize(fv)
        return fv





#%%

if __name__=='__main__':
    #Opening TF session and building the net
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, None, None, 3])
    network = fvcnn(imgs, 'vgg16_weights.npz', sess)

    #Reading images
    img_folder = "./dtd/images/*/"
    label_folder = "./dtd/labels/"
    extns = "*.jpg"

    #Create dictionary with filenames/labels
    data_labels = {}
    label_list = {}
    files = []
    file_labels = []
    index = 0
    with open('./dtd/labels/labels_joint_anno.txt') as f:
        for line in f:
            words = line.split()
            filename = "./dtd/images/"+ str(words[0])
            filelabel = line.split("/")[0]
            if(not filelabel in label_list):
                label_list[filelabel] = index
                index += 1
            data_labels[filename] = label_list[filelabel] #Uncomment for int labels in dictionary
            files.append(filename)
            file_labels.append(label_list[filelabel])
            #data_labels[filename = filelabel #Uncomment for strings labels in dictionary

    #Form train/test set
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    pics_to_load = None
    current_fold = 0
    max_folds=10

    # SELECT HERE WHICH FOLDS YOU WANT TO RUN, IN ASCENDING ORDER (due to zero-indent, from 0 to 9)
    # FOLD 0: MIRZA
    # FOLD 1: MIRZA
    # FOLD 2: MIRZA
    # FOLD 3: LORENZO
    # FOLD 4: LORENZO
    # FOLD 5: JESPER
    # FOLD 6: JESPER
    # FOLD 7: SOREN
    # FOLD 8: SOREN
    # FOLD 9: SOREN

    to_do_folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for train, test in kf.split(files, file_labels):
        # LOOP OVER LESS FOLDS?
        if(not (current_fold in to_do_folds)):
            current_fold += 1
            continue

        print("#################################################")
        print("#                 NEXT FOLD, "+str(current_fold)+"                  #")
        print("#################################################")
        # We use the mat_descripts to convert the features of each image into a fv,
        # the stacked descripts to learn the gmm and the loaded file tracker to
        # write the labels
        #Since we're using lists, we have some pre-processing to do
        train_data = [files[i] for i in train]
        test_data = [files[i] for i in test]

        #Shuffle the data
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)

        print("\n-> Loading/propagating training pics through the CNN...\n")
        train_mat_descripts, train_descripts, train_loaded_paths = network.forward_pass(train_data, pics_to_load)
        print("Complete! \n")

        print("\n-> Loading/propagating test pics through the CNN...\n")
        test_mat_descripts, test_descripts, test_loaded_paths = network.forward_pass(test_data, pics_to_load)
        print("-> Complete! \n")

        train_loaded_labels = []
        for i in range(len(train_loaded_paths)):
            train_loaded_labels.append(data_labels[train_loaded_paths[i]])

        test_loaded_labels = []
        for i in range(len(test_loaded_paths)):
            test_loaded_labels.append(data_labels[test_loaded_paths[i]])
    #%%
        N = 64
        print("-> Dictionary creation through GMM...")
        network.generate_GMM(train_descripts, N)
        print("-> ...completed! \n")

        print("-> Computing training FV")
        train_fv = []
        for i in range(len(train_mat_descripts)):
            print("\t - Computing FV for pic "+str(i) + "/" +str(len(train_mat_descripts)))
            train_fv.append(network.fisher_vector(np.concatenate(np.concatenate(train_mat_descripts[i]))))
        train_fv = np.stack(train_fv)
        print("Done! \n")

        print("-> Computing test FV")
        test_fv = []
        for i in range(len(test_mat_descripts)):
            print("\t - Computing FV for pic "+str(i) + "/" +str(len(train_mat_descripts)))
            test_fv.append(network.fisher_vector(np.concatenate(np.concatenate(test_mat_descripts[i]))))
        test_fv = np.stack(test_fv)
        print("-> Done! \n")

        #SVM classify
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
        results = clf.predict(test_fv)

        #%%
        #Saving the workspace

        #Extracting network parameters that we want to save
        means = network.means
        covs = network.covs
        weights = network.weights

        filename = "fold" + str(current_fold) + "_results"
        current_directory = os.getcwd()
        final_directory = os.path.join(current_directory, filename)
        if not os.path.exists(final_directory):
            os.makedirs(filename)
        os.chdir(final_directory)

        import pickle
        for key in dir():
            try:
                to_save = globals()[key]
                if(isinstance(to_save, ModuleType)):
                    continue
                f = open(key,"wb")
                pickle.dump(to_save,f)
                f.close()
            except:
                pass

        os.chdir(current_directory)

        current_fold += 1

#%%

        ################ IGNORE THIS ##################



#    ##UNCOMMENT TO RESTORE WORKSPACE
#    import os
#    import shelve
#    filename = "fold0_results"
#    current_directory = os.getcwd()
#    final_directory = os.path.join(current_directory, filename)
#    if not os.path.exists(final_directory):
#        os.makedirs(filename)
#    os.chdir(final_directory)
#
#
#    my_shelf = shelve.open(filename)
#    vari = []
#    for key in my_shelf:
#        vari.append(key)
#
#    for key in vari:
#        globals()[key]=my_shelf[key]
#    #print(str(key))
#    my_shelf.close()
#
#    os.chdir(current_directory)
#
#import pickle
#import os
#import shelve
#f = open("train_fv","rb")
#train_fv = pickle.load(f)
#f.close()