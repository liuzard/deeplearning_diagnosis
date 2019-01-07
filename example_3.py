import tensorflow as tf
a=tf.Variable(tf.truncated_normal(shape=[1,2],dtype=tf.float32))
b=tf.concat([a,a],axis=1)
tf.layers.max_pooling1d()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(b))
