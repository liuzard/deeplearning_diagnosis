import tensorflow as tf
import numpy as np

alpha=0.01
is_train=True
batch_noise = tf.constant(1.0,dtype=tf.float32,shape=(64,1024))


layer1 = tf.layers.dense(batch_noise, 64* 256)
layer1 = tf.reshape(layer1, [-1, 64, 256])
# batch normalization
layer1 = tf.layers.batch_normalization(layer1, training=is_train)
# Leaky ReLU
layer1 = tf.maximum(alpha * layer1, layer1)
# dropout
layer1 = tf.nn.dropout(layer1, keep_prob=0.8)

print(layer1)

# 4 x 4 x 512 to 7 x 7 x 256
filter1=tf.constant(1.0,shape=[5,128,256])
layer2=tf.contrib.nn.conv1d_transpose(layer1,filter1,output_shape=[64,128,128],stride=2,padding="SAME")
# layer2 = tf.layers.conv2d_transpose(layer1, 256, 4, strides=2, padding='SAME')
layer2 = tf.layers.batch_normalization(layer2, training=is_train)
layer2 = tf.nn.leaky_relu(layer2,alpha=alpha)
layer2 = tf.nn.dropout(layer2, keep_prob=0.8)


# 256x1x256 to 512x1x128
filter2=tf.constant(1.0,shape=[5,64,128])
layer3=tf.contrib.nn.conv1d_transpose(layer2,filter2,output_shape=[64,256,64],stride=2,padding='SAME')
# layer3 = tf.layers.conv2d_transpose(layer2, 128, 3, strides=2, padding='SAME')
layer3 = tf.layers.batch_normalization(layer3, training=is_train)
layer3 = tf.nn.leaky_relu(layer3,alpha=alpha)
layer3 = tf.nn.dropout(layer3, keep_prob=0.8)


# 512x1x128 to 1024 x 1x1
filter3=tf.constant(1.0,shape=[11,1,64])
logits=tf.contrib.nn.conv1d_transpose(layer3,filter3,output_shape=[64,1024,1],stride=4,padding='SAME')
# logits = tf.layers.conv2d_transpose(layer3, output_dim, 3, strides=2, padding='SAME')
# MNIST原始数据集的像素范围在0-1，这里的生成图片范围为(-1,1)
# 因此在训练时，记住要把MNIST像素范围进行resize
outputs = tf.tanh(logits)
print(outputs)


# y5=tf.constant(-1.0)
# x1=tf.constant(1.0,shape=[10,3,4])
# kernel=tf.constant(1.0,shape=[3,5,4])
# x2=tf.constant(1.0,shape=[2,6,4])
# x3=tf.constant(1.0,shape=[2,5,4])
# # y1=tf.nn.conv1d(x2,kernel,stride=2,padding="SAME")
# y2=tf.contrib.nn.conv1d_transpose(x1,kernel,output_shape=[10,6,5],stride=2,padding="SAME")
# # print(y1)
# print(y2)
#
# with tf.Session() as  sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(tf.nn.leaky_relu(y2)))