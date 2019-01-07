# -*-coding utf-8 -*-
#--------CHAPTER05 MNIST最佳样例实践之二：训练模块------------------#
import os
import numpy as np
import pandas as pd
import sklearn.preprocessing as pre
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data


from BPNetwork_MNIST import BPNetwork_MNIST_inference

BATCH_SIZE=100
BAES_LEARNING_RATE=0.5
LEARNING_RATE_DECAY=0.99
AVERAGE_MOVING_DECAY=0.99
REGULARZITION_RATE=0.00001
TRAINING_STEPS=1000000

MODEL_SAVE_PATH="E:\DeepLearning_model\BPNetwork\BPNetwork_MNIST"
MODEL_NAME="BPNetwork_MNIST.ckpt"

def train(mnist):
    x=tf.placeholder(dtype=tf.float32, shape=[None, BPNetwork_MNIST_inference.INPUT_NODE], name="input")
    y_=tf.placeholder(dtype=tf.float32, shape=[None, BPNetwork_MNIST_inference.OUTPUT_NODE], name="labels")
    regularizer=tf.contrib.layers.l1_regularizer(REGULARZITION_RATE)
    y= BPNetwork_MNIST_inference.inference(x, regularizer)
    global_step=tf.Variable(0,trainable=False)

    variable_average=tf.train.ExponentialMovingAverage(AVERAGE_MOVING_DECAY,global_step)
    average_op=variable_average.apply(tf.trainable_variables())
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
    learing_rate=tf.train.exponential_decay(BAES_LEARNING_RATE,global_step,mnist.train.num_examples / BATCH_SIZE,LEARNING_RATE_DECAY,staircase=True)

    train_step=tf.train.GradientDescentOptimizer(learing_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,average_op]):
        # train_op=tf.no_op(name='train')
        train_op = tf.no_op()

    saver=tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            if i%100==0:
                print("After %d training steps,the loss of the model is %g"%(step,loss_value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)


def main(argv=None):
    mnist=input_data.read_data_sets('/tmp/data',one_hot=True)
    train(mnist)

if __name__ == '__main__':
    main()