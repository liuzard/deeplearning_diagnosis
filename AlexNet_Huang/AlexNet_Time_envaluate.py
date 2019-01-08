# -*- coding :utf-8 -*-
'''
1、构建AlexNet
2、计算前向传播时间
3、计算后向传播时间
'''
# --------导入用到的库----------
from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batch = 100



def print_activations(t):
    print(t.op.name, '', t.get_shape().as_list())


def inference(images):
    parameters = []
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32, stddev=0.1), name="weights")
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        conv1 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]

    lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool1")
    print_activations(pool1)

    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], stddev=0.1, dtype=tf.float32), name="weights")
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
        conv = tf.nn.conv2d(lrn1, kernel, [1, 1, 1, 1], padding="SAME")
        conv2 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)

        print_activations(conv2)
        parameters += [kernel, biases]

    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool2")
    print_activations(pool2)

    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 192, 384], stddev=0.1, dtype=tf.float32), name="weights")
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name="biases")
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding="SAME")
        conv3 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)

        print_activations(conv3)
        parameters += [kernel, biases]

    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=0.1, dtype=tf.float32), name="weights")
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name="biases")
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding="SAME")
        conv4 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)
        print_activations(conv4)
        parameters += [kernel, biases]

    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1, dtype=tf.float32), name="weights")
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name="biases")
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding="SAME")
        conv5 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)
        print_activations(conv5)
        parameters += [kernel, biases]

    pool3 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool3")
    print_activations(pool3)
    return pool3, parameters


def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_square = 0.0

    for i in range(num_steps_burn_in + num_batch):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration=%.3f' % (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_square += duration * duration
    mn = total_duration / num_batch
    vr = total_duration_square / num_batch - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps,%.3f +/- %.3f sec /batch' % (datetime.now(), info_string, num_batch, mn, sd))


def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3], dtype=tf.float32, stddev=0.1))
        pool5, parameters = inference(images)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        time_tensorflow_run(sess, pool5, 'Forward')
        objective = tf.nn.l2_loss(pool5)
        grad = tf.gradients(objective, parameters)
        time_tensorflow_run(sess, grad, "Forward-Backward")


def main(argv=None):
    run_benchmark()


if __name__ == "__main__":
    tf.app.run()
