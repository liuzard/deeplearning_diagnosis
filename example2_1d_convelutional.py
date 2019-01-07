import tensorflow as tf
from tensorflow.contrib import slim
input=tf.get_variable("input",shape=[4,10,1],initializer=tf.truncated_normal_initializer(stddev=0.1),dtype=tf.float32)
filter=tf.ones([3, 1, 5])
# pool=tf.get_variable("pool_input",[4,2,5,16],initializer=tf.truncated_normal_initializer(stddev=0.1),dtype=tf.float32)
output=tf.nn.conv1d(input,filter,stride=2,padding="SAME")
pool=tf.layers.max_pooling1d(output,pool_size=2,strides=2,padding="SAME")
# pool=tf.nn.max_pool(pool,ksize=[1,2,3,1],strides=[1,1,1,1],padding="VALID")


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("output",sess.run(output))
    print("pool:",sess.run(pool))
