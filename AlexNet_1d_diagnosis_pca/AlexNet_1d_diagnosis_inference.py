# -*- coding: utf-8 -*-
# ---------AlexNet卷积神经网络前向传播模块-------------------

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
FC3_WIDTH = 10


def inference(input_tensor, train, regularizer):
    with tf.variable_scope("layer1-conv1"):
        filter = tf.get_variable("filter", [CONV1_WIDTH, IMAGE_CHANNELS, CONV1_DEEP],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1), dtype=tf.float32)

        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv1d(input_tensor, filter, stride=3, padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # with tf.name_scope("layer2-lrn1"):
    #     lrn1=tf.nn.lrn(relu1,2,bias=1.0,alpha=0.001/9,beta=0.75)
    with tf.name_scope("layer3-pool1"):
        pool1 = tf.layers.max_pooling1d(relu1, pool_size=2, strides=2, padding='SAME')

    with tf.variable_scope("layer4-conv2"):
        filter = tf.get_variable("filter", [CONV2_WIDTH, CONV1_DEEP, CONV2_DEEP],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1), dtype=tf.float32)

        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv1d(pool1, filter, stride=2, padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # with tf.name_scope("layer5-lrn2"):
    #     lrn2=tf.nn.lrn(relu2,2,bias=1.0,alpha=0.001/9,beta=0.75)


    with tf.name_scope("layer6-pool2"):
        pool2 = tf.layers.max_pooling1d(relu2, pool_size=2, strides=2, padding='SAME')

    with tf.variable_scope("layer7-conv3"):
        filter = tf.get_variable("filter", [CONV3_WIDTH, CONV2_DEEP, CONV3_DEEP],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1), dtype=tf.float32)

        conv3_biases = tf.get_variable("bias", [CONV3_DEEP], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv1d(pool2, filter, stride=1, padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    with tf.variable_scope('layer8-conv4'):
        filter = tf.get_variable("filter", [CONV4_WIDTH, CONV3_DEEP, CONV4_DEEP],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1), dtype=tf.float32)

        conv4_biases = tf.get_variable("bias", [CONV4_DEEP], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv1d(relu3, filter, stride=1, padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

    with tf.variable_scope("layer9-conv5"):
        filter = tf.get_variable("filter", [CONV5_WIDTH, CONV4_DEEP, CONV5_DEEP],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1), dtype=tf.float32)

        conv5_biases = tf.get_variable("bias", [CONV5_DEEP], initializer=tf.constant_initializer(0.0))
        conv5 = tf.nn.conv1d(relu4, filter, stride=1, padding='SAME')
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_biases))

    with tf.name_scope("layer10-pool3"):
        pool3 = tf.layers.max_pooling1d(relu5, pool_size=2, strides=2, padding='SAME')

    pool_shape = pool3.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2]
    print(nodes)
    reshaped = tf.reshape(pool3, [pool_shape[0], nodes])

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
        logit = tf.matmul(fc2, fc3_weights) + fc3_biase
        # logit=tf.nn.softmax(tf.matmul(fc2,fc3_weights)+fc3_biase)
        print(logit)
    return fc2,logit

