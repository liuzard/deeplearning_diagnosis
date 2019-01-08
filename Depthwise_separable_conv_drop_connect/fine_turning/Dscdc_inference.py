# -*- coding: utf-8 -*-
# ---------Depth_sep卷积神经网络前向传播模块-------------------

# 1、导入需要用到的库
import tensorflow as tf

# 2、配置卷积神经网络参数
INPUT_NODE = 2000
OUTPUT_NODE = 10

IMAGE_WIDTH = 2000
IMAGE_CHANNELS = 1
IMAGE_LABELS = 10

# 第一层卷积层的尺寸和深度
CONV1_WIDTH = 20
CONV1_DEEP = 16
CONV1_STRIDE=9

# 第二层卷积层的尺寸和深度（深度可分卷积）
CONV2_WIDTH = 11
CONV2_DEEP = 32
CONV2_STRIDE=5

# 第三层卷积层的尺寸和深度
CONV3_WIDTH = 5
CONV3_DEEP = 64
CONV3_STRIDE=3

# 全连接层网络参数
# FC1_WIDTH = 500
FC2_WIDTH = 100
FC3_WIDTH = 10


def inference(input_tensor, train, regularizer):
    with tf.variable_scope("layer1-conv1"):
        filter = tf.get_variable("filter", [CONV1_WIDTH, IMAGE_CHANNELS, CONV1_DEEP],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1), dtype=tf.float32)

        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv1d(input_tensor, filter, stride=3, padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.layers.max_pooling1d(relu1, pool_size=2, strides=2, padding='SAME')

    with tf.variable_scope("layer3-depth_sep_conv1"):
        sep_conv1 = tf.layers.separable_conv1d(pool1, CONV2_DEEP, CONV2_WIDTH, strides=CONV2_STRIDE, padding="SAME")
        relu2=tf.nn.relu(sep_conv1)

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.layers.max_pooling1d(relu2, pool_size=2, strides=1, padding='SAME')

    with tf.variable_scope("layer5-depth_sep_conv2"):
        sep_conv2 = tf.layers.separable_conv1d(pool2, CONV3_DEEP, CONV3_WIDTH, strides=CONV3_STRIDE, padding="SAME")
        relu2=tf.nn.relu(sep_conv2)

    conv_shape = relu2.get_shape().as_list()
    nodes = conv_shape[1] * conv_shape[2]
    reshaped = tf.reshape(relu2, [conv_shape[0], nodes])

    with tf.variable_scope('layer7-fc2'):
        fc2_weights = tf.get_variable("weights", [nodes, FC2_WIDTH],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_biases = tf.get_variable("bias", [FC2_WIDTH], initializer=tf.constant_initializer(0.1))
        if regularizer != None:
            tf.add_to_collection("losses", regularizer(fc2_weights))
        if train:
            fc2_weights = tf.nn.dropout(fc2_weights, 0.5)
        fc2 = tf.nn.relu(tf.matmul(reshaped, fc2_weights) + fc2_biases)

    with tf.variable_scope('layer8-fc3'):
        fc3_weights = tf.get_variable("weight", [FC2_WIDTH, FC3_WIDTH],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc3_biase = tf.get_variable("bias", [FC3_WIDTH], initializer=tf.constant_initializer(0.1))
        if regularizer != None:
            tf.add_to_collection("losses", regularizer(fc3_weights))
        if train:
            fc3_weights = tf.nn.dropout(fc3_weights, 0.5)
        logit = tf.matmul(fc2, fc3_weights)+fc3_biase
    return logit
