from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from multi_perceptron import multi_perceptron

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
mnist = input_data.read_data_sets('/tmp/data', one_hot=True)
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


X_train, X_test =mnist.train.images, mnist.test.images

n_samples = int(mnist.train.num_examples)
training_epochs =10
batch_size = 128
display_step = 1



with tf.name_scope("autoencoder1"):
    mp1 =multi_perceptron (
        n_input=784,
        n_hidden=500,
        n_output=10,
        transfer_function=tf.nn.tanh,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
        )
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)

            # Fit training using batch data
            cost = mp1.partial_fit(batch_xs,batch_ys)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%d,' % (epoch + 1),
                  "Cost:", "{:.9f}".format(avg_cost))
            print("Total cost: " + str(mp1.calc_accuracy(mnist.test.images, mnist.test.labels)))

# with tf.name_scope("autoencoder2"):
#     mp1 =multi_perceptron (
#         n_input=500,
#         n_hidden=200,
#         n_output=10,
#         transfer_function=tf.nn.tanh,
#         optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
#         )
#     for epoch in range(training_epochs):
#         avg_cost = 0
#         total_batch = int(n_samples / batch_size)
#         # Loop over all batches
#         for i in range(total_batch):
#             batch_xs,batch_ys = mnist.train.next_batch(batch_size)
#
#             # Fit training using batch data
#             cost = mp1.partial_fit(batch_xs,batch_ys)
#             # Compute average loss
#             avg_cost += cost / n_samples * batch_size
#
#         # Display logs per epoch step
#         if epoch % display_step == 0:
#             print("Epoch:", '%d,' % (epoch + 1),
#                   "Cost:", "{:.9f}".format(avg_cost))
#             print("Total cost: " + str(mp1.calc_accuracy(mnist.test.images, mnist.test.labels)))

