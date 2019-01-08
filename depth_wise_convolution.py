import tensorflow as tf
import time

image = tf.constant(value=1, shape=[10, 1000, 16], dtype=tf.float32)
filter = tf.get_variable("filter", [5, 16, 32],
                         initializer=tf.truncated_normal_initializer(stddev=0.1), dtype=tf.float32)
output_sep = tf.layers.separable_conv1d(image, 32, 5, strides=3, padding="VALID")
output_nor = tf.nn.conv1d(image, filter, stride=3, padding="VALID")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    time_start = time.time()
    for i in range(10000):
        output_nor_value = sess.run(output_nor)
    time_end = time.time()
    time_use_nor = time_end - time_start
    print("nor_use:", time_use_nor)
    time_start=time.time()
    for i in range(10000):
        output_sep_value = sess.run(output_sep)
    time_end=time.time()
    time_use_sep=time_end-time_start
    print("sep_use:",time_use_sep)


