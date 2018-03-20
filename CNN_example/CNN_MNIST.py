### EXAMPLE TAKEN FROM THE DEEP LEARNING COURSE, WEEK 2

#This example uses the tensorflow framework to define the Network structure. However,
#instead of using the layers definition provided by tensorflow, the layers provided by the keras
#wrapper are used. There is no sensible difference in the outcome, at least afaik.

from __future__ import absolute_import, division, print_function 

import matplotlib.pyplot as plt
import numpy as np
# import sklearn.datasets
import tensorflow as tf
import os
import sys
sys.path.append(os.path.join('.', '..')) 
import utils 

#%%
## IMPORTING THE DATASET

# Load data (download if you haven't already)
from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('MNIST_data', 
                                       one_hot=True,   # Convert the labels into one hot encoding
                                       dtype='float32', # rescale images to `[0, 1]`
                                       reshape=False, # Don't flatten the images to vectors
                                      )

## Print dataset statistics and visualize
print('')
utils.mnist_summary(mnist_data)

#%%

##BUILDING UP THE NETWORK

from keras.layers import Input, Dense, Conv2D, MaxPooling2D
from tensorflow.contrib.layers import flatten # We use this flatten, as it works better than 
                                              # the Keras 'Flatten' for some reason
###TWO LAYER CNN###
tf.reset_default_graph()

num_classes = 10
height, width, nchannels = 28, 28, 1

padding_1 = 'same'
filters_1 = 8
kernel_size_1 = (4,4)
pool_size_1 = (7,7)

padding_2 = 'same'
filters_2 = 8
kernel_size_2 = (height//pool_size_1[0],width//pool_size_1[1])
pool_size_2 = (4,4)

x_pl_1 = tf.placeholder(tf.float32, [None, height, width, nchannels], name='xPlaceholder_1')
y_pl_1 = tf.placeholder(tf.float32, [None, None, None, None], name = 'yPlaceholder_1')
x_pl_2 = tf.placeholder(tf.float32, [None, None, None, None], name='xPlaceholder_2')
y_pl_2 = tf.placeholder(tf.float32, [None, num_classes], name='yPlaceholder_2')


print('Trace of the tensors shape as it is propagated through the network.')
print('Layer name \t Output size')
print('----------------------------')

with tf.variable_scope('convLayer1'):
   conv1 = Conv2D(filters_1, kernel_size_1, strides=(1,1), padding=padding_1, activation='relu')
   print('x_pl_1 \t\t', x_pl_1.get_shape())
   x = conv1(x_pl_1)
   print('conv1 \t\t', x.get_shape())

   pool1 = MaxPooling2D(pool_size=pool_size_1, strides=None, padding=padding_1)
   x = pool1(x)
   print('pool1 \t\t', x.get_shape())
   #x = flatten(x)
   #print('Flatten \t', x.get_shape())
   
with tf.variable_scope('convLayer2'):
   conv2 = Conv2D(filters_2, kernel_size_2, strides=(1,1), padding=padding_2, activation='relu')
   #print('x_pl_2 \t\t', x_pl_2.get_shape())
   x2 = conv2(x)
   print('conv2 \t\t', x2.get_shape())

   pool2 = MaxPooling2D(pool_size=pool_size_2, strides=None, padding=padding_1)
   x2 = pool2(x2)
   print('pool2 \t\t', x2.get_shape())
   x2 = flatten(x2)
   print('Flatten \t', x2.get_shape())


with tf.variable_scope('output_layer'):
   denseOut = Dense(units=num_classes, activation='softmax')
   
   y = denseOut(x2)
   print('denseOut\t', y.get_shape())    

print('Model consits of ', utils.num_params(), 'trainable parameters.')


#%%

##DEFINE LOSS/TRAINING/PERFORMANCE OPERATORS

with tf.variable_scope('loss'):
    # computing cross entropy per sample
    cross_entropy = -tf.reduce_sum(y_pl_2 * tf.log(y+1e-8), reduction_indices=[1])

    # averaging over samples
    cross_entropy = tf.reduce_mean(cross_entropy)

    
with tf.variable_scope('training'):
    # defining our optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

    # applying the gradients
    train_op = optimizer.minimize(cross_entropy)

    
with tf.variable_scope('performance'):
    # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
    correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pl_2, axis=1))

    # averaging the one-hot encoded vector
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
#%%
    
## TESTING THE FORWARD PASS
    
#Test the forward pass
x_batch, y_batch = mnist_data.train.next_batch(4)
gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=1)


with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:
# with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_pred = sess.run(fetches=y, feed_dict={x_pl_1: x_batch})

assert y_pred.shape == y_batch.shape, "ERROR the output shape is not as expected!" \
        + " Output shape should be " + str(y.shape) + ' but was ' + str(y_pred.shape)

print('Forward pass successful!')

#%%

## TRAINING

#Training Loop
batch_size = 100
max_epochs = 10


valid_loss, valid_accuracy = [], []
train_loss, train_accuracy = [], []
test_loss, test_accuracy = [], []


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Begin training loop')

    try:
        while mnist_data.train.epochs_completed < max_epochs:
            _train_loss, _train_accuracy = [], []
            
            ## Run train op
            x_batch, y_batch = mnist_data.train.next_batch(batch_size)
            fetches_train = [train_op, cross_entropy, accuracy]
            feed_dict_train = {x_pl_1: x_batch, y_pl_2: y_batch}
            _, _loss, _acc = sess.run(fetches_train, feed_dict_train)
            
            _train_loss.append(_loss)
            _train_accuracy.append(_acc)
            

            ## Compute validation loss and accuracy
            if mnist_data.train.epochs_completed % 1 == 0 \
                    and mnist_data.train._index_in_epoch <= batch_size:
                train_loss.append(np.mean(_train_loss))
                train_accuracy.append(np.mean(_train_accuracy))

                fetches_valid = [cross_entropy, accuracy]
                
                feed_dict_valid = {x_pl_1: mnist_data.validation.images, y_pl_2: mnist_data.validation.labels}
                _loss, _acc = sess.run(fetches_valid, feed_dict_valid)
                
                valid_loss.append(_loss)
                valid_accuracy.append(_acc)
                print("Epoch {} : Train Loss {:6.3f}, Train acc {:6.3f},  Valid loss {:6.3f},  Valid acc {:6.3f}".format(
                    mnist_data.train.epochs_completed, train_loss[-1], train_accuracy[-1], valid_loss[-1], valid_accuracy[-1]))
        
        
        test_epoch = mnist_data.test.epochs_completed
        while mnist_data.test.epochs_completed == test_epoch:
            x_batch, y_batch = mnist_data.test.next_batch(batch_size)
            feed_dict_test = {x_pl_1: x_batch, y_pl_2: y_batch}
            _loss, _acc = sess.run(fetches_valid, feed_dict_test)
            test_loss.append(_loss)
            test_accuracy.append(_acc)
        print('Test Loss {:6.3f}, Test acc {:6.3f}'.format(
                    np.mean(test_loss), np.mean(test_accuracy)))


    except KeyboardInterrupt:
        pass

#%%
        
##PLOTTING
        
epoch = np.arange(len(train_loss))
plt.figure()
plt.plot(epoch, train_accuracy,'r', epoch, valid_accuracy,'b')
plt.legend(['Train Acc','Val Acc'], loc=4)
plt.xlabel('Epochs'), plt.ylabel('Acc'), plt.ylim([0.75,1.03])