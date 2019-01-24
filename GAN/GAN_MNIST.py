import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(r"C:\Users\liuzard\PycharmProjects\deep_learning_diagnosis\MNIST_data")


# 1、定义输入
def get_input(real_size, noise_size):
    """
    :param real_size: 真实图像尺寸
    :param noise_size: 输入噪声尺寸
    :return: 返回真实图片和噪声的placeholder
    """
    real_img = tf.placeholder(tf.float32, shape=[None, real_size], name="real_img")
    noise = tf.placeholder(tf.float32, shape=[None, noise_size], name="noise")
    return real_img, noise


# 2、定义生成器

def generator(noise, hidden_dim, out_dim, reuse=False, alpha=0.01):
    """
    :param noise: 输入噪声
    :param hidden_dim: 隐层维度
    :param out_dim: 输出维度
    :param reuse: 复用性
    :param alpha: leaky ReLU 参数
    :return: 返回生成的参数
    """
    with tf.Variable("generator", reuse=reuse):
        hidden1 = tf.layers.dense(noise, hidden_dim)
        hidden1 = tf.nn.leaky_relu(hidden1, alpha=alpha)
        hidden1 = tf.layers.dropout(hidden1, rate=0.2)
        logits = tf.layers.dense(hidden1, out_dim)
        outputs = tf.tanh(logits)
        return logits, outputs


# 3、定义判别器
def discriminator(image, hidden_dim, reuse=False, alpha=0.01):
    """
    :param image: 输入图像，包含真假图像
    :param hidden_dim: 隐层维度
    :param out_dim: 输出维度
    :param reuse: 复用性
    :param alpha: leaky ReLU 参数
    :return: 返回判别的结果
    """
    with tf.Variable("discriminator", reuse=reuse):
        # hidden layer
        hidden1 = tf.layers.dense(image, hidden_dim)
        hidden1 = tf.nn.leaky_relu(hidden1, alpha=alpha)

        # output layer
        logits = tf.layers.dense(hidden1, 1)
        outputs = tf.sigmoid(logits)
        return logits, outputs


# 4、定义参数
real_size = 784
noise_size = 100
hidden_gen = 128
hidden_dis = 128
learning_rate = 0.001
smooth = 0.1

# 5、构建网络
tf.reset_default_graph()
real_img, noise = get_input(real_size, noise_size)
# 生成器
g_logits, g_outouts = generator(noise=noise, hidden_dim=hidden_gen, out_dim=real_size)

# 判别器，包含两部分：判断真实的图片和判断生成的图片
d_logits_real, d_outputs_real = discriminator(image=real_img, hidden_dim=hidden_gen)
d_logits_fake, d_outputs_fake = discriminator(image=g_outouts, hidden_dim=hidden_dis, reuse=True)

# 6、损失函数
d_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(d_logits_real, tf.ones_initializer(d_logits_real))) * (1 - smooth)
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_logits_fake, tf.zeros_like(d_logits_fake)))
d_loss=tf.add(d_loss_fake+d_loss_real)


g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_logits_fake, tf.ones_like(d_logits_fake))) * (
            1 - smooth)

# 7、training
train_vars = tf.trainable_variables()

# generator 中的tensor
g_vars = [var for var in train_vars if var.name.startswith("generator")]
d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

# generator 和 discirminator 的优化器
d_train_opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(d_loss,var_list=d_vars)
g_train_opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(g_loss,var_list=g_vars)


