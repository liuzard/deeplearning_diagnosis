import tensorflow as tf

# ---------------定义网络输入和输出参数-------------------
IMAGE_SIZE = 32
IMAGE_CHANNELS = 1
IMAGE_LABELS = 9

# ---------------定义网络参数----------------------
CONV1_SIZE = 5
CONV1_DEEP = 32

CONV2_SIZE = 5
CONV2_DEEP = 64

FC_NODES = 512


# -------------前向传播函数------------------------
def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        weights = tf.get_variable("weights", [CONV1_SIZE, CONV1_SIZE, IMAGE_CHANNELS, CONV1_DEEP],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, weights, strides=[1, 1, 1, 1], padding="SAME")
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, biases))

    with tf.variable_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    with tf.variable_scope("layer3-conv2"):
        weights = tf.get_variable("weights", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
        initializer = tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, weights, strides=[1, 2, 2, 1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, biases))

    with tf.variable_scope("layer3-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        pool2_shape = pool2.get_shape().as_list()
    nodes = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]
    fc_nodes = tf.reshape(pool2, [pool2_shape[0], nodes])

    with tf.variable_scope("layer5-fc1"):
        weights = tf.get_variable("weights", [nodes, FC_NODES], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [FC_NODES], initializer=tf.constant_initializer(0.0))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(weights))
        fc1 = tf.nn.relu(tf.matmul(fc_nodes, weights) + biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'):
        weights = tf.get_variable("weights", [FC_NODES, IMAGE_LABELS],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [IMAGE_LABELS], initializer=tf.constant_initializer(0.0))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(weights))
        fc2 = tf.matmul(fc1, weights) + biases
        if train:
            fc2 = tf.nn.dropout(fc2, 0.5) + biases
    return fc1,fc2