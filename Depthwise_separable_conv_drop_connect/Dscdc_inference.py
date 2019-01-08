# -*- coding: utf-8 -*-
# ---------AlexNet卷积神经网络前向传播模块-------------------

# 1、导入需要用到的库
import tensorflow as tf

# 2、配置卷积神经网络参数
INPUT_NODE = 2000
OUTPUT_NODE = 9

IMAGE_WIDTH = 1024
IMAGE_CHANNELS = 1
IMAGE_LABELS = 9

# 第一层卷积层的尺寸和深度
CONV1_WIDTH = 20
CONV1_DEEP = 16

# 第二层卷积层的尺寸和深度
CONV2_WIDTH = 11
CONV2_DEEP = 32

# 第三层卷积层的尺寸和深度
CONV3_WIDTH = 5
CONV3_DEEP = 64

# 第四层卷积层尺寸和深度
CONV4_WIDTH = 5
CONV4_DEEP = 128

# 第五层卷积层尺寸和深度
CONV5_WIDTH = 3
CONV5_DEEP = 128

# 全连接层网络参数
FC1_WIDTH = 500
FC2_WIDTH = 100
FC3_WIDTH = 9


def inference(input_tensor, train, regularizer):
    with tf.variable_scope("layer1-conv1"):
        filter = tf.get_variable("filter", [CONV1_WIDTH, IMAGE_CHANNELS, CONV1_DEEP],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1), dtype=tf.float32)

        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv1d(input_tensor, filter, stride=3, padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # with tf.name_scope("layer2-lrn1"):
    #     lrn1=tf.nn.lrn(relu1,2,bias=1.0,alpha=0.001/9,beta=0.75)
    with tf.name_scope("layer2-pool1"):
        pool1 = tf.layers.max_pooling1d(relu1, pool_size=2, strides=2, padding='SAME')

    with tf.variable_scope("layer3-conv2"):
        filter = tf.get_variable("filter", [CONV2_WIDTH, CONV1_DEEP, CONV2_DEEP],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1), dtype=tf.float32)

        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv1d(pool1, filter, stride=2, padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # with tf.name_scope("layer5-lrn2"):
    #     lrn2=tf.nn.lrn(relu2,2,bias=1.0,alpha=0.001/9,beta=0.75)


    with tf.name_scope("layer4-pool2"):
        pool2 = tf.layers.max_pooling1d(relu2, pool_size=2, strides=1, padding='SAME')



    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2]
    print(nodes)
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    with tf.variable_scope("layer11-fc1"):
        fc1_wieghts = tf.get_variable("weight", [nodes, FC1_WIDTH],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc1_biases = tf.get_variable("bias", [FC1_WIDTH], initializer=tf.constant_initializer(0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_wieghts))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_wieghts) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer12-fc2'):
        fc2_weights = tf.get_variable("weights", [FC1_WIDTH, FC2_WIDTH],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_biases = tf.get_variable("bias", [FC2_WIDTH], initializer=tf.constant_initializer(0.1))
        if regularizer != None:
            tf.add_to_collection("losses", regularizer(fc2_weights))
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train:
            fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('layer13-fc3'):
        fc3_weights = tf.get_variable("weight", [FC2_WIDTH, FC3_WIDTH],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc3_biase = tf.get_variable("bias", [FC3_WIDTH], initializer=tf.constant_initializer(0.1))
        if regularizer != None:
            tf.add_to_collection("losses", regularizer(fc3_weights))
        logit = tf.nn.softmax(tf.matmul(fc2, fc3_weights)+fc3_weights)
    return fc2,logit
