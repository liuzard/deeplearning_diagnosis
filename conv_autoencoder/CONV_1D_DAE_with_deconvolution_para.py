'''
1d convolutional autoencoder
1、with deconvolutional
2、

@author:liuzard
date:2018-3-15
'''

#import module
import tensorflow as tf
import numpy as np
import math
from conv_autoencoder import matfile_reader


def get_random_block_form_data(data_sampls,batch_size):
    start_index=np.random.randint(0,len(data_sampls)-batch_size)
    return data_sampls[start_index:(start_index+batch_size)]

#load data
DATA_PATH="E:\\bearing_dataset_1024.mat"
data=matfile_reader.dataset_reader(DATA_PATH)
data=data[:,0:1024]
data=data.reshape(-1,1024,1)
noise_factor =0.1
epochs = 5
batch_size = 128


#network para
INPUT_DM=1024
SIGNAL_CHANNELS=1

#first convolutional layer para
CONV1_WIDTH=20
CONV1_DEPTH=16

#second convolutional layer para
CONV2_WIDTH=11
CONV2_DEPTH=32

#third convoltional layer para
CONV3_WIDTH=3
CONV3_DEPTH=64

#first deconvolutional layer para
DECONV1_WIDTH=3
DECONV1_DEPTH=64

#second deconvoltional layer para
DECONV2_WIDTH=11
DECONV2_DEPTH=32

#third deconvolutional layer para
DECONV3_WIDTH=20
DECONV3_DEPTH=1

#train para
batch_size=128

inputs_ = tf.placeholder(tf.float32, (None,1024, 1), name='inputs_')
targets_ = tf.placeholder(tf.float32, (None, 1024, 1), name='targets_')


with tf.variable_scope("layer1-conv1"):
    filter1 = tf.get_variable("filter1", [CONV1_WIDTH, SIGNAL_CHANNELS, CONV1_DEPTH],
                             initializer=tf.truncated_normal_initializer(stddev=0.1), dtype=tf.float32)

    conv1_biases = tf.get_variable("bias", [CONV1_DEPTH], initializer=tf.constant_initializer(0.0))
    print(conv1_biases)
    conv1 = tf.nn.conv1d(inputs_, filter1, stride=3, padding='SAME')
    relu1 = tf.nn.tanh(tf.nn.bias_add(conv1, conv1_biases))
print(relu1)
with tf.variable_scope("layer2-conv2"):
    filter2 = tf.get_variable("filter2", [CONV2_WIDTH, CONV1_DEPTH, CONV2_DEPTH],
                             initializer=tf.truncated_normal_initializer(stddev=0.1), dtype=tf.float32)

    conv2_biases = tf.get_variable("bias", [CONV2_DEPTH], initializer=tf.constant_initializer(0.0))
    conv2 = tf.nn.conv1d(relu1, filter2, stride=3, padding='SAME')
    relu2 = tf.nn.tanh(tf.nn.bias_add(conv2, conv2_biases))

with tf.variable_scope("layer3-conv3"):
    filter3 = tf.get_variable("filter3", [CONV3_WIDTH, CONV2_DEPTH, CONV3_DEPTH],
                             initializer=tf.truncated_normal_initializer(stddev=0.1), dtype=tf.float32)

    conv3_biases = tf.get_variable("bias", [CONV3_DEPTH], initializer=tf.constant_initializer(0.0))
    conv3= tf.nn.conv1d(relu2, filter3, stride=3, padding='SAME')
    relu3 = tf.nn.tanh(tf.nn.bias_add(conv3, conv3_biases))
print(relu3)

with tf.variable_scope("layer4-deconv1"):
    filter4 = tf.get_variable(name="filter4", shape=[DECONV1_WIDTH, DECONV1_DEPTH, CONV3_DEPTH],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    # deconv1_biases=tf.get_variable("bias", [DECONV1_DEPTH], initializer=tf.constant_initializer(0.0))
    relu4 = tf.contrib.nn.conv1d_transpose(relu3, filter4, output_shape=[batch_size, 114, DECONV1_DEPTH], stride=3, padding="SAME")
    # relu4=tf.nn.bias_add(deconv1,deconv1_biases)
    # layer2 = tf.layers.conv2d_transpose(layer1, 256, 4, strides=2, padding='SAME')
    # layer2 = tf.nn.tanh(layer2)

with tf.variable_scope("layer5-deconv2"):
    filter5 = tf.get_variable(name="filter5", shape=[DECONV2_WIDTH, DECONV2_DEPTH, DECONV1_DEPTH],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    # deconv2_biases = tf.get_variable("bias", [DECONV2_DEPTH], initializer=tf.constant_initializer(0.0))
    relu5 = tf.contrib.nn.conv1d_transpose(relu4, filter5, output_shape=[batch_size, 342, DECONV2_DEPTH], stride=3, padding="SAME")
    # relu5 = tf.nn.bias_add(deconv2, deconv2_biases)

with tf.variable_scope("layer6-deconv3"):
    filter6 = tf.get_variable(name="filter6", shape=[DECONV3_WIDTH, DECONV3_DEPTH, DECONV2_DEPTH],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    # deconv3_biases=tf.get_variable("bias", [DECONV3_DEPTH], initializer=tf.constant_initializer(0.0))
    outputs_ = tf.contrib.nn.conv1d_transpose(relu5, filter6, output_shape=[batch_size, 1024, DECONV3_DEPTH], stride=3, padding="SAME")
    # layer2 = tf.layers.conv2d_transpose(layer1, 256, 4, strides=2, padding='SAME')
    # outputs_=tf.nn.bias_add(deconv3,deconv3_biases)
print(outputs_)

loss =tf.squared_difference(outputs_,targets_)
    # 0.5*tf.pow(tf.subtract(outputs_,targets_),2)
cost=tf.reduce_sum(loss)

optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for idx in range(data.shape[0]// batch_size):
            signals = get_random_block_form_data(data,batch_size)
            signal_sum=np.sum(np.power(signals,2))
            print(signal_sum)
            # 加入噪声
            noisy_imgs = signals + noise_factor * np.random.randn(*signals.shape)
            b=0.2*np.random.randn(*signals.shape)
            noise_sum=np.sum(np.power(b,2))
            SNR=10*math.log10(signal_sum/noise_sum)
            print(SNR)
            batch_cost, _ = sess.run([cost, optimizer],
                                     feed_dict={inputs_: noisy_imgs,
                                                targets_: signals})

            print("Epoch: {}/{} ".format(e + 1, epochs),
                  "Training loss: {:.4f}".format(batch_cost))