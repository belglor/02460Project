########################################################################################
# VGG16 implementation in TensorFlow taken and adapted from http://www.cs.toronto.edu/~frossard/post/vgg16/                                                  
# Copyright Davi Frossard, 2016                                                                            
#                                     
# Fisher Vector implementation taken and adapted from https://github.com/jacobgil/pyfishervector                                                                                
#
########################################################################################

import tensorflow as tf
import numpy as np
import imageio
from skimage import transform
import sys
import glob
import math
from sklearn import mixture
from scipy.stats import multivariate_normal

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
    
    # Fit a GMM to descriptors (HOW ARE DESCRIPTORS FED?)
    def generate_GMM(self, descriptors, N):
#        em = cv2.ml.EM_create()
#        em.setClustersNumber(N)
#        em.trainEM(descriptors)
        gmm = mixture.GaussianMixture(N)
        gmm.fit(descriptors)
        return np.float32(gmm.means_), \
        		np.float32(gmm.covariances_), np.float32(gmm.weights_)
                
    # Compute moments of gaussian distribution 
    def likelihood_moment(self, x, ytk, moment):	
        	x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
        	return x_moment * ytk
    
    # Given a GMM model (given by means, covs and mixture weights) and samples, compute
    # the samples statistics wro the given GMM
    def likelihood_statistics(self, samples, means, covs, weights):
        n_samples = samples.shape[0]
        dim_samples = samples.shape[1]
        n_mixtures = weights.shape[0]
        gaussians = np.zeros([n_samples, n_mixtures])
        s0, s1, s2 = np.zeros([n_mixtures]), np.zeros([dim_samples, n_mixtures]), np.zeros([dim_samples, n_mixtures])
        samples = zip(range(0, len(samples)), samples)
        #BE CAREFUL: I'M ALLOWING SINGULAR MATRICES FOR COVARIANCE
        g = [multivariate_normal(mean=means[k], cov=covs[k], allow_singular = True) for k in range(0, weights.size) ]
        for index, x in samples:
            gaussians[index] = np.array([g_k.pdf(x) for g_k in g])
        # Set inf or NaN probs to 1
        for j in range(gaussians.shape[0]):
            for k in range(gaussians.shape[1]):                
                if(math.isnan(gaussians[j,k]) or math.isinf(gaussians[j,k]) or (gaussians[j,k]>1)):
                    gaussians[j,k] = 1
                        
        for k in range(0, weights.size):
            s0[k], s1[k], s2[k] = 0, 0, 0
            for index, x in samples:
                probabilities = np.multiply(gaussians[index], weights.T)                
                probabilities = probabilities / np.sum(probabilities)
                s0[k] = s0[k] + self.likelihood_moment(x, probabilities[k], 0)
                s1[:,k] = s1[:,k] + self.likelihood_moment(x, probabilities[k], 1)
                s2[k] = s2[k] + self.likelihood_moment(x, probabilities[k], 2)

        return s0, s1, s2
    
    def fisher_vector_weights(self, s0, s1, s2, means, covs, w, T):
        	return np.float32([((s0[k] - T * w[k]) / np.sqrt(w[k]) ) for k in range(0, len(w))])
    
    def fisher_vector_means(self, s0, s1, s2, means, sigma, w, T):
        	return np.float32([(s1[:,k] - means[k] * s0[k]) / (np.sqrt(w[k] * sigma[k])) for k in range(0, len(w))])
    
    def fisher_vector_sigma(self, s0, s1, s2, means, sigma, w, T):
        	return np.float32([(s2[:,k] - 2 * means[k]*s1[:,k]  + (means[k]*means[k] - sigma[k]) * s0[k]) / (np.sqrt(2*w[k])*sigma[k])  for k in range(0, len(w))])
    
    def normalize(self, fisher_vector):
        	v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
        	return v / np.sqrt(np.dot(v, v))
    
    # Compute a sample's Fisher Vector
    def fisher_vector(self, samples, means, covs, weights):
        s0, s1, s2 =  self.likelihood_statistics(samples, means, covs, weights)
        T = samples.shape[0]
        covs = np.float32([np.diagonal(covs[k]) for k in range(0, covs.shape[0])])
        a = self.fisher_vector_weights(s0, s1, s2, means, covs, weights, T)
        b = self.fisher_vector_means(s0, s1, s2, means, covs, weights, T)
        c = self.fisher_vector_sigma(s0, s1, s2, means, covs, weights, T)
        fv = np.concatenate([np.concatenate(a), np.concatenate(b), np.concatenate(c)])
        fv = self.normalize(fv)
        return fv
        
    # Propagate an image through the model:
    # THIS HAS TO BE MODIFIED TO TAKE INTO ACCOUNT TRAINING OR JUST PREDICTING
    # When training, a GMM should be fitted to he given data (passed as stacked up images)
    # When predicting, this is not necessary and the image is just propagated through the
    # CNN , the first and second order statistic is calculated wro the fitted GMM and then
    # classification through SVM is performed
    
    # The method takes a collection of images with dimension [N_pics, 224, 224, 3] and the
    # current tf session as input. The images are propagated through the convolutional layers
    # of the pre treained VGG16. Then, the corresponding FV are extracted. Finally, the FV
    # are classified through a SVM
    def forward_propagate(self, imgs, sess):
        # Propagate images through CNN: extract descriptors
        mat_descripts = sess.run(self.descripts, feed_dict={self.imgs: imgs})
        #Concatenate descriptors to have (7*7*N_images)x(512*N_images)
        descripts = mat_descripts[0,:,:,:]
        for i in range(mat_descripts.shape[0]-1):
            descripts = np.concatenate((descripts, mat_descripts[i+1,:,:,:]), axis=2)
        descripts = np.concatenate(descripts)
        #Cluster with GMM: use EM and return components means, covs and weights
        #Number of GMM components
        N = 64 
        means, covs, weights = self.generate_GMM(descripts.T, N) #NB THERE HAS TO BE SOME FORM OF RESHAPING/CONCATENATION
        #Throw away gaussians with weights that are too small: TO FIX
#        th = 1.0 / N
#        means = np.float32([m for k,m in zip(range(0, len(weights)), means) if weights[k] > th])
#        covs = np.float32([m for k,m in zip(range(0, len(weights)), covs) if weights[k] > th])
#        weights = np.float32([m for k,m in zip(range(0, len(weights)), weights) if weights[k] > th])

        #Compute FV
        fv = []
        for i in range(mat_descripts.shape[0]):
            fishvec = self.fisher_vector(np.concatenate(mat_descripts[i,:,:,:]), means, covs, weights)
            fv.append(fishvec)
        return fv
        #Classify with SVM
    
#%%

#Opening TF session and building the net        
sess = tf.Session()
imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
network = fvcnn(imgs, 'vgg16_weights.npz', sess)

#Reading images
folder = "./dtd/images/*/"
extns = "*.jpg"
files = glob.glob(folder+extns)
batch_size = 5
batch_to_load = 1 # Use len(files)//batch_size for whole dataset
feed_imgs = np.zeros([batch_size, 224, 224, 3])
loaded_tracker = [] #tracker to not random sample same pic twice
print('Dividing the data in ' + str(batch_to_load) + ' batches of size ' + str(batch_size) + ':')
#Propagate through CNN in batches
for i in range(batch_to_load): 
    print('Loading batch #' +str(i))
    sys.stdout.flush() 
    #Stacking the next batch together
    for j in range(batch_size):
        #If not using the dataset, take random pics
        if(batch_to_load != len(files)//batch_size):
            index = np.random.randint(0, len(files)) 
            while(index in loaded_tracker): #check if duplicated entry
                index = np.random.randint(0, len(files)) 
            loaded_tracker.append(index) #track loaded pics
        else:
            index = j + i*batch_size
        #Check index to prevent out-of-bounds
        if(index>=len(files)):
            break
        img = imageio.imread(files[index])
        img = transform.resize(img, (224, 224))
        feed_imgs[j+i*batch_size,:,:,:] = img
    #Feeding the stacked batch to the CNN
    mat_descripts = sess.run(network.descripts, feed_dict={network.imgs: feed_imgs})
    #Concatenate descriptors to have (7*7*N_images)x(512)
    if(i==0):
        descripts = np.concatenate(mat_descripts[0,:,:,:])
        for j in range(mat_descripts.shape[0]-1):
            descripts = np.concatenate((descripts, np.concatenate(mat_descripts[j+1,:,:,:])))
    else:
        tmp_descripts = np.concatenate(mat_descripts[0,:,:,:])
        for j in range(mat_descripts.shape[0]-1):
            tmp_descripts = np.concatenate((tmp_descripts, np.concatenate(mat_descripts[j+1,:,:,:])))
        descripts = np.concatenate((descripts, tmp_descripts))
#%%
#Cluster with GMM: use EM and return components means, covs and weights
#Number of GMM components
N = 64
means, covs, weights = network.generate_GMM(descripts, N) #NB THERE HAS TO BE SOME FORM OF RESHAPING/CONCATENATION

#%%
#Compute FV
fv = []
for i in range(mat_descripts.shape[0]):
    #FV procedure (unrolled for debugging)
    samples = np.concatenate(mat_descripts[i,:,:,:])
    # SHOULD BE FURTHER UNROLLED: LOOK AT THE GENERAL METHOD DESCRIPTIONS IN THE fvcnn CLASS
    s0, s1, s2 =  network.likelihood_statistics(samples, means, covs, weights)
    T = samples.shape[0]
    tmp_covs = np.float32([np.diagonal(covs[k]) for k in range(0, covs.shape[0])])
    a = network.fisher_vector_weights(s0, s1, s2, means, tmp_covs, weights, T)
    b = network.fisher_vector_means(s0, s1, s2, means, tmp_covs, weights, T)
    c = network.fisher_vector_sigma(s0, s1, s2, means, tmp_covs, weights, T)
    fishvec = np.concatenate([a, np.concatenate(b), np.concatenate(c)])
    fishvec = network.normalize(fishvec)
    fv.append(fishvec)

# Close tf session
sess.close()
#
#
