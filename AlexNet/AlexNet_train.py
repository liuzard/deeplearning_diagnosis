import os

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from AlexNet import AlexNet_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.05
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.001
TRAINING_STEPS = 20000
MOVING_AVERAGE_DECAY = 0.99
SAVE_PATH="E:/MNIST/ALexNet"
MODEL_NAME="LetNet5_mnist_model.ckpt"


def train(mnist):
    # 定义输出为4维矩阵的placeholder
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,
        AlexNet_inference.IMAGE_SIZE,
        AlexNet_inference.IMAGE_SIZE,
        AlexNet_inference.IMAGE_CHANNELS],
                       name='x-input')

    y_ = tf.placeholder(tf.float32, [None, AlexNet_inference.IMAGE_LABELS], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = AlexNet_inference.inference(x, False, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')


    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                AlexNet_inference.IMAGE_SIZE,
                AlexNet_inference.IMAGE_SIZE,
                AlexNet_inference.IMAGE_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            if i % 100 == 0:
                print("After %d training step(s), loss on training batch is %g" % (step, loss_value))
                saver.save(sess, os.path.join(SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    main()