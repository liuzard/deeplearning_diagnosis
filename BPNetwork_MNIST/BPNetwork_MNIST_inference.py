# -*-coding:utf-8 -*-
#-----------Chapertor 05 MNIST最佳实践样例之一：前向传播程序--------------------#
import tensorflow as tf
INPUT_NODE=784
HIDDEN_NODE=200
OUTPUT_NODE=10

def get_weight_variable(shape,regularizer):
    weights=tf.get_variable("weights",shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer!=None:
        tf.add_to_collection('losses',regularizer(weights))
    return weights

def inference(inputtensor,regularizer):
    with tf.variable_scope("layer1"):
        weights=get_weight_variable([INPUT_NODE,HIDDEN_NODE],regularizer)
        biases=tf.get_variable("biases",shape=[HIDDEN_NODE],initializer=tf.zeros_initializer())
        # layer1=tf.nn.relu(tf.matmul(inputtensor,weights)+biases)
        layer1=tf.layers.dense(inputtensor,HIDDEN_NODE,use_bias=True,activation=tf.nn.relu)
    with tf.variable_scope("layer2"):
        weights=get_weight_variable([HIDDEN_NODE,OUTPUT_NODE],regularizer)
        biases=tf.get_variable("biases",shape=[OUTPUT_NODE],initializer=tf.zeros_initializer())
        # layer2=tf.matmul(layer1,weights)+biases
        layer2=tf.layers.dense(layer1,OUTPUT_NODE,use_bias=True)
    return layer2


