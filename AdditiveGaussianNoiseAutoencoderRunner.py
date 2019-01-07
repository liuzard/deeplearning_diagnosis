from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

from autoencoder_models.DenoisingAutoencoder import AdditiveGaussianNoiseAutoencoder

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
mnist = input_data.read_data_sets('/tmp/data', one_hot=True)


def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


X_train, X_test =mnist.train.images, mnist.test.images

n_samples = int(mnist.train.num_examples)
training_epochs =10
batch_size = 128
display_step = 1



with tf.name_scope("autoencoder1"):
    autoencoder1 = AdditiveGaussianNoiseAutoencoder(
        n_input=784,
        n_hidden=1000,
        transfer_function=tf.nn.tanh,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.0001),
        scale=0.1)
    print(autoencoder1.weights)
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, batch_size)

            # Fit training using batch data
            cost = autoencoder1.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%d,' % (epoch + 1),
                  "Cost:", "{:.9f}".format(avg_cost))

    print("Total cost: " + str(autoencoder1.calc_total_cost(X_test)))
    X_reconstruct = autoencoder1.reconstruct(X_train)


with tf.variable_scope("autoencoder2"):
    autoencoder2 = AdditiveGaussianNoiseAutoencoder(
        n_input=1000,
        n_hidden=1000,
        transfer_function=tf.nn.tanh,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.0001),
        scale=0.1)

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_reconstruct, batch_size)

            # Fit training using batch data
            cost = autoencoder1.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%d,' % (epoch + 1),
                  "Cost:", "{:.9f}".format(avg_cost))

    print("Total cost: " + str(autoencoder1.calc_total_cost(X_test)))
    X_reconstruct2 = autoencoder1.reconstruct(X_reconstruct)

with tf.name_scope('autoencoder3'):
    autoencoder3 = AdditiveGaussianNoiseAutoencoder(
        n_input=1000,
        n_hidden=500,
        transfer_function=tf.nn.tanh,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.0001),
        scale=0.1)

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_reconstruct2, batch_size)

            # Fit training using batch data
            cost = autoencoder1.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%d,' % (epoch + 1),
                  "Cost:", "{:.9f}".format(avg_cost))

    print("Total cost: " + str(autoencoder1.calc_total_cost(X_test)))


print(autoencoder1.getWeights())

N_INPUT=784
N_HIDDEN=1000
N_HIDDEN_2=1000
N_HIDDEN_3=500
N_OUTPUT=10
train_epoch=300

x=tf.placeholder(dtype=tf.float32,shape=[None,N_INPUT],name="x-input")
y_=tf.placeholder(dtype=tf.float32,shape=[None,N_OUTPUT],name="y-input")
with tf.name_scope("fc_layer1"):
    w1=tf.Variable(autoencoder1.getWeights(),trainable=False)
    bias1=tf.Variable(tf.zeros([N_HIDDEN],dtype=tf.float32))
    hidden=tf.nn.sigmoid(tf.matmul(x,w1)+bias1)

with tf.name_scope('fc2_layer2'):
    # w2 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=(N_HIDDEN, N_HIDDEN_2),stddev=0.1))
    w2 = tf.Variable(autoencoder2.getWeights(),trainable=False)
    biase2 = tf.Variable(tf.zeros([N_HIDDEN_2], dtype=tf.float32))
    hidden2=tf.nn.sigmoid(tf.matmul(hidden,w2)+biase2)

with tf.name_scope('fc3_layer3'):
    # w3=tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=(N_HIDDEN_2, N_HIDDEN_3),stddev=0.1))
    w3 = tf.Variable(autoencoder3.getWeights())
    biase3=tf.Variable(tf.zeros([N_HIDDEN_3]),dtype=tf.float32)
    hidden3=tf.nn.sigmoid(tf.matmul(hidden2,w3)+biase3)

with tf.name_scope('fc4_layer4'):
    w4=tf.Variable(tf.truncated_normal(dtype=tf.float32,shape=(N_HIDDEN_3,N_OUTPUT),stddev=0.1),name="w4")
    biase4=tf.Variable(tf.zeros(N_OUTPUT),dtype=tf.float32)
    y=tf.matmul(hidden3,w4)+biase4

loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
loss_mean=tf.reduce_mean(loss)
train_op=tf.train.GradientDescentOptimizer(0.5).minimize(loss_mean)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_loop=int(mnist.train.num_examples/batch_size)
    for i in range(train_epoch):
        for j in range(num_loop):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run([train_op,accuracy],feed_dict={x:batch_xs,y_:batch_ys})
        if i%1==0:
            accuracy_value=sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
            print("after %d training steps ,the accuracy in train dataset is %f" %(i,accuracy_value))


# batch_xs=get_random_block_from_data(X_train,10)




# print(autoencoder.getWeights())




'''
X_reconstruct=autoencoder.reconstruct(batch_xs)

# 原始图片和重构图片可视化
f,a=plt.subplots(2,10,figsize=(10,2))
for i in range(10):
    a[0][i].imshow(np.reshape(batch_xs[i], (28, 28)))
    a[1][i].imshow(np.reshape(X_reconstruct[i], (28, 28)))
f.show()
plt.savefig('example_ad_tanh.png')

print(X_reconstruct.shape)
'''

