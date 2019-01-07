# -*- coding: utf-8 -*-
# ---------AlexNet卷积神经网络前向传播模块-------------------

# 1、导入需要用到的库
import tensorflow as tf

# 2、配置卷积神经网络参数
INPUT_NODE = 2000
OUTPUT_NODE = 10

IMAGE_SIZE = 32
IMAGE_CHANNELS = 1
IMAGE_LABELS = 10

# 第一层卷积层的尺寸和深度
CONV1_SIZE = 7
CONV1_DEEP = 16

# 第二层卷积层的尺寸和深度
CONV2_SIZE = 5
CONV2_DEEP = 32

# 第三层卷积层的尺寸和深度
CONV3_SIZE = 3
CONV3_DEEP = 64

# 第四层卷积层尺寸和深度
CONV4_SIZE = 3
CONV4_DEEP = 128

# 第五层卷积层尺寸和深度
CONV5_SIZE = 3
CONV5_DEEP = 256

# 全连接层网络参数
FC1_SIZE = 500
FC2_SIZE = 100
FC3_SIZE = 10


def inference(input_tensor, train, regularizer):
    with tf.variable_scope("layer1-conv1"):
        conv1_weights = tf.get_variable("weihts", [CONV1_SIZE, CONV1_SIZE, IMAGE_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 3, 3, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-lrn1"):
        lrn1 = tf.nn.lrn(relu1, 4, bias=1.0, alpha=0.001 / 9, beta=0.75)
    with tf.name_scope("layer3-pool1"):
        pool1 = tf.nn.max_pool(lrn1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("layer4-conv2"):
        conv2_weights = tf.get_variable("weihts", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer5-lrn2"):
        lrn2 = tf.nn.lrn(relu2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75)

    with tf.name_scope("layer6-pool2"):
        pool2 = tf.nn.max_pool(lrn2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("layer7-conv3"):
        conv3_weights = tf.get_variable("weihts", [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [CONV3_DEEP], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    with tf.variable_scope('layer8-conv4'):
        conv4_weights = tf.get_variable("weihts", [CONV4_SIZE, CONV4_SIZE, CONV3_DEEP, CONV4_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [CONV4_DEEP], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(relu3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

    with tf.variable_scope("layer9-conv5"):
        conv5_weights = tf.get_variable("weihts", [CONV5_SIZE, CONV5_SIZE, CONV4_DEEP, CONV5_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv5_biases = tf.get_variable("bias", [CONV5_DEEP], initializer=tf.constant_initializer(0.0))
        conv5 = tf.nn.conv2d(relu4, conv5_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_biases))

    with tf.name_scope("layer10-pool3"):
        pool3 = tf.nn.max_pool(relu5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    pool_shape = pool3.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    print(nodes)
    reshaped = tf.reshape(pool3, [pool_shape[0], nodes])

    with tf.variable_scope("layer11-fc1"):
        fc1_wieghts = tf.get_variable("weight", [nodes, FC1_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc1_biases = tf.get_variable("bias", [FC1_SIZE], initializer=tf.constant_initializer(0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_wieghts))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_wieghts) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer12-fc2'):
        fc2_weights = tf.get_variable("weights", [FC1_SIZE, FC2_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_biases = tf.get_variable("bias", [FC2_SIZE], initializer=tf.constant_initializer(0.1))
        if regularizer != None:
            tf.add_to_collection("losses", regularizer(fc2_weights))
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train:
            fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('layer13-fc3'):
        fc3_weights = tf.get_variable("weight", [FC2_SIZE, FC3_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc3_biase = tf.get_variable("bias", [FC3_SIZE], initializer=tf.constant_initializer(0.1))
        if regularizer != None:
            tf.add_to_collection("losses", regularizer(fc3_weights))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biase

    return fc2,logit