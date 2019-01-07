import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 训练参数
learning_rate=0.01
learning_rate_train=0.8
pre_num_epochs=10
train_num_epochs=100
batch_size=100
display_step=10
example_to_show=10

#网络参数
num_input=784
num_hidden=200
num_construct=num_input
num_out=10

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
mnist=input_data.read_data_sets('/tmp/data',one_hot=True)

def WeightsVariable(n_in,n_out,name_str):
    return tf.Variable(tf.random_normal([n_in,n_out],stddev=0.1),dtype=tf.float32,name=name_str)

def BiasesVariable(n_bias,name_str):
    return tf.Variable(tf.zeros([n_bias]),dtype=tf.float32,name=name_str)

def encoder(x_origin,act_fun=tf.nn.tanh):
    with tf.name_scope('layer'):
        weights=WeightsVariable(num_input,num_hidden,'enc_weights')
        biases=BiasesVariable(num_hidden,'enc_biases')
        x_hidden=act_fun(tf.matmul(x_origin,weights)+biases)
        return x_hidden


def decoder(x_hidden,act_fun=tf.nn.tanh):
    with tf.name_scope('layer'):
        weights=WeightsVariable(num_hidden,num_construct,'dec_weights')
        biases=BiasesVariable(num_construct,'dec_biases')
        x_construct=act_fun(tf.matmul(x_hidden,weights)+biases)
        return x_construct

with tf.Graph().as_default():
    with tf.name_scope('input'):
        x=tf.placeholder(dtype=tf.float32,shape=[None,num_input],name='input')
    with tf.name_scope('encoder'):
        x_coder=encoder(x)
    with tf.name_scope('decoder'):
        x_decoder=decoder(x_coder)
    with tf.name_scope('loss_ae'):
        loss_ae=tf.reduce_mean(tf.pow(x_decoder-x,2))
    with tf.name_scope('pre_train'):
        pre_train=tf.train.AdamOptimizer(learning_rate).minimize(loss_ae)
    with tf.name_scope('fc'):
        droup_out_rate=tf.placeholder(tf.float32)
        weights=tf.Variable(tf.truncated_normal([num_hidden,num_out],stddev=0.1),dtype=tf.float32,name='weights_fc')
        biase=tf.Variable(tf.zeros([num_out]),dtype=tf.float32,name='biases_fc')
        y_original=tf.nn.softmax(tf.matmul(x_coder,weights)+biase)
        y=tf.nn.dropout(y_original,keep_prob=droup_out_rate)
    with tf.name_scope('train'):
        y_=tf.placeholder(dtype=tf.float32,shape=[None,10],name='y-input')
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        loss= tf.reduce_mean(cross_entropy)
        train=tf.train.GradientDescentOptimizer(learning_rate_train).minimize(loss)
    with tf.name_scope('evaluate'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init=tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        train_num=int(mnist.train.num_examples/batch_size)
        for i in range(pre_num_epochs):
            for j in range(train_num):
                xs,ys=mnist.train.next_batch(batch_size)
                _,loss_value=sess.run([pre_train,loss_ae],feed_dict={x:xs})
            print(i+1,'=',loss_value)
        print("pre_train finish")

        for p in range(train_num_epochs):
            for q in range(train_num):
                xs,ys=mnist.train.next_batch(batch_size)
                _f,loss_train=sess.run([train,loss],feed_dict={x:xs,y_:ys,droup_out_rate:0.5})
            print(p+1,"=",loss_train)
            xs_vd,ys_vd=mnist.validation.images,mnist.validation.labels
            accuracy_value=sess.run(accuracy,feed_dict={x:xs_vd,y_:ys_vd,droup_out_rate:1})
            print(accuracy_value)
        print("train finished")



        reconstruction=sess.run([x_decoder],feed_dict={x:mnist.test.images[0:example_to_show]})
        print(reconstruction[0])



#原始图片和重构图片可视化
        f,a=plt.subplots(2,10,figsize=(10,2))
        for i in range(example_to_show):
            a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
            a[1][i].imshow(np.reshape(reconstruction[0][i], (28, 28)))
        f.show()
        plt.savefig('example.png')