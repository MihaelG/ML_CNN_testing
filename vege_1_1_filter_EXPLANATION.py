#!/usr/bin/env python
# coding: utf-8

# # Training of CNN with 1x1 filter 
# 
# There are already some papers where classification of vegetation is made on images using CNN. CNN is used primarily for classification on images where segments of an image are spatially dependet. I belive that results could be better with using pixel based classification.
# 
# Why using 1x1 filters?
# 
# The idea is to make algorithm to find connections between values in channels of pixels (red, green, blue etc.) and corresponding output (e.g. is it vegetation or not).
# 
# Tips while testing algorithm:
# - activation functon - Elu and ReLU act. functions were tested (in conv layers), Elu gave better results
# - normalisation - is there any good reason to use normalisation since it is useful primarily in very deep nets, and this one is deep.
# - regularization - is there reason to use regularisation in CNN??? - think not...
# - optimizer - ADAM? Usually Adam is the best solution, but try others as well...
# - try using different cost functions
# - try using more then 1 fully connected layer
# 
# 
# Viktor Mihoković master thesis:
# Paper where classification of images was made with CNN. The goal of the alogrithm was to classify vegetation and non-vegetation on images made by GoPro camera. 
# In that camera lens for detecting red wavelengths was replaces with lens for detecting IR wavelengths.....

# ## Dataset
# Dataset used in this algorithm is .csv file where each row represents one pixel. First column is the name of the class of the instance (i.e. pixel), and other columns are values of different channels of the instance.

# In[1]:


import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf
import csv

from sklearn.metrics import accuracy_score

# Common imports
import numpy as np
import os

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
import time


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


print ('DATA PREPARATION')
vege_csv = csv.reader(open('vegetacija.csv', newline=''), delimiter=' ', quotechar='|')

vege_array = []


for row in vege_csv:
	red_array = []
	items = row[0].split(';')
	for i in items:
		red_array.append(i)

	vege_array.append(red_array)


klase = []
values = []

vege_array_fix = vege_array[:-4]

print ('length of the fixed array is ', len (vege_array_fix))

for r in vege_array_fix:
    value_row = []
    i = 0
    while i < len(r):
        if i == 0:
            if r[i] == 'X':
                klase.append(0)
            if r[i] == 'V':
                klase.append(1)
            if r[i] == 'N':
                klase.append(2)
        else:
            value_row.append(float(r[i])/1000)
        i = i + 1
    values.append(value_row)

print ('test : ', values[10000])


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(values, klase, test_size=0.20)


# In[4]:


reset_graph()


# In[5]:


height = 1
width = 1
channels = 8
# n_inputs = height * width

conv1_fmaps = 32
conv1_ksize = 1
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 1
conv2_stride = 1
conv2_pad = "SAME"

conv3_fmaps = 128
conv3_ksize = 1
conv3_stride = 1
conv3_pad = "SAME"

conv4_fmaps = 128
conv4_ksize = 1
conv4_stride = 1
conv4_pad = "SAME"

# conv5_fmaps = 256
# conv5_ksize = 1
# conv5_stride = 1
# conv5_pad = "SAME"

# conv6_fmaps = 512
# conv6_ksize = 1
# conv6_stride = 1
# conv6_pad = "SAME"


# In[6]:


pool5_fmaps = conv4_fmaps

#making slef made filters (maybe will be used sometimes later)
# filters = np.zeros(shape=(1, 1, channels), dtype=np.float32)
# import random
# for i in range(0, channels):
#   filters[:, :, i] = (random.randint(20,101))

n_outputs = 3

# n_inputs = len(X_train)

with tf.name_scope("inputs"):
    #scope names are defined just to make TF graf easier to comprehend
    X = tf.placeholder(tf.float32, shape=[None, channels], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name="y")

conv1 = tf.layers.conv2d(X_reshaped, filters =conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.elu, name="conv1")
'''
X_reshaped - input for the first convolutional layer
filters = ..., - number of filters in convolution (dimensionality of the output space)
kernel_size = integer (or tuple/list of 2 integers), specifying the height and width of 
2D convolutional window

strides = -An integer or tuple/list of 2 integers, specifying the strides of the convolution 
along the height and width. Can be a single integer to specify the same value for all spatial 
dimensions

padding - same or valid

activation - activation fuction, if it is set to None that it will maintain a linear activation.
In this case elu or ReLU activation functions are good options. With few testing elu activation
function gave better results thus I have choose it.
'''
conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.elu, name="conv2")
conv3 = tf.layers.conv2d(conv2, filters=conv3_fmaps, kernel_size=conv3_ksize,
                         strides=conv3_stride, padding=conv3_pad,
                         activation=tf.nn.elu, name="conv3") 
conv4 = tf.layers.conv2d(conv3, filters=conv4_fmaps, kernel_size=conv4_ksize,
                         strides=conv4_stride, padding=conv4_pad,
                         activation=tf.nn.elu, name="conv4") 

# conv5 = tf.layers.conv2d(conv4, filters=conv5_fmaps, kernel_size=conv5_ksize,
#                          strides=conv5_stride, padding=conv5_pad,
#                          activation=tf.nn.elu, name="conv5")

# conv6 = tf.layers.conv2d(conv5, filters=conv6_fmaps, kernel_size=conv6_ksize,
#                          strides=conv6_stride, padding=conv6_pad,
#                          activation=tf.nn.elu, name="conv6")


# In[7]:


with tf.name_scope("pool5"):
    pool5 = tf.nn.avg_pool(conv4, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding="VALID")
    pool5_flat = tf.reshape(pool5, shape=[-1, pool5_fmaps * 1 * 1])
'''
tf.nn.avg_pool(....
conv4 - value, a 4D tensor
ksize - a list or tuple of 4 ints. The size of the window for each dimension of the input tensor.
strides - a list or tuple of 4 ints. The stride of the sliding window for each dimension of the input tensor.

DID SWITCHING BETWEEN avg_pool and max_pool MADE ANY CHANGES IN ACCURACY?
It gave 88.6% (0.7% less) accuracy, but it converged faster.

'''
n_fc1 = 128
# n_fc2 = 256

with tf.name_scope("fc"):
    fc1 = tf.layers.dense(pool5_flat, n_fc1, activation=tf.nn.elu, name="fc1")
#     fc2 = tf.layers.dense(pool5_flat, n_fc2, activation=tf.nn.relu, name="fc2")
  
'''
tf.layers.dense - used for defining a fully connected layer
pool7_flat - tensor input
n_fc1 - units, Integer or Long, dimensionality of the output space
'''    
with tf.name_scope("output"):
    logits = tf.layers.dense(fc1, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")
    
'''
LOGITS is fully connected layer, but with output with number of classes to predict. 
Logits are unnormalized final scores of model. Softmax activation function is used to get 
probability distributions over classes.
'''

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
# sparse_softmax_cross_entropy_with_logits - cost function used in this learning
# logits - input values for cost function, unscaled log probabilities of shape [d_0, d_1, ..., d_{r-1}, 
# num_classes] and dtype float16, float32, or float64
    loss = tf.reduce_mean(xentropy)

#tf.reduce_mean - Computes the mean of elements across dimensions of a tensor

    optimizer = tf.train.AdamOptimizer()
#chosen optimizer here is Adam and it is used in minimazing cost function
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
# tf.nn.in_top_k - says whether the targets are in top k predictions
# logits - predictions: A Tensor of type float32. A batch_size x classes tensor.
# y - targets 
# 1 - k: number of top elements to look for computing precision (must be int)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
#tf.cast - casts a tensor to a new type

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


# In[8]:


X_valid = X_train[:63808]
X_train = X_train[63808:]

y_valid = y_train[:63808]
y_train = y_train[63808:]


# In[9]:


train_array_length = len(X_train)
print (train_array_length)


# In[10]:



n_epochs = 1500
batch_size = 50

max_checks_without_progress = 500
checks_without_progress = 0
best_loss = np.infty

start = time.time()

with tf.Session() as sess:
    init.run()

    for epoch in range(n_epochs):
        rnd_idx = np.random.permutation(len(X_train))
        for rnd_indices in np.array_split(rnd_idx, len(X_train) // batch_size):
            X_batch = []
            y_batch = []
            for i in rnd_indices:
                X_batch.append(X_train[i]) 
                y_batch.append(y_train[i])
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
# Here we run a session with optimizer where we feed session with X and y of a batch size
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X: X_valid, y: y_valid})
# With running previous session loss and accuracy for validation dataset is calculated.
        if loss_val < best_loss:
            save_path = saver.save(sess, "./vege_model.ckpt")
#saving model and its parameters with saver.save, so it can be used later
            best_loss = loss_val
            checks_without_progress = 0
        else:
            checks_without_progress += 1
            if checks_without_progress > max_checks_without_progress:
                print("Early stopping!")
                break
        print("{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".format(
            epoch, loss_val, best_loss, acc_val * 100))

with tf.Session() as sess:
    saver.restore(sess, "./vege_model.ckpt")
    acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
    print("Final test accuracy: {:.2f}%".format(acc_test * 100))
end = time.time()
print ('[ EXECUTION TIME ] =', (end - start))   


# In[ ]:




