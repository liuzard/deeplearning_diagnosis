import os
import numpy as np
import sklearn.preprocessing as pre
import tensorflow as tf
import time

from Depthwise_separable_conv_drop_connect import Dscdc_inference
from Depthwise_separable_conv_drop_connect import matfile_reader

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 60000
MOVING_AVERAGE_DECAY = 0.99
SAVE_PATH=r"G:\06 深度学习模型\depthsie_separable_dropconnect"
MODEL_NAME="depthsie_separable_dropconnect.ckpt"



def one_hot_transform(formalarray,input_label):#把原始编码转换成One-hot编码
    enc=pre.OneHotEncoder()
    enc.fit(formalarray)
    return enc.transform(input_label).toarray()

def data_preprocess(data):#对原始数据进行处理，得到样本和样本标签
    data=np.array(data)
    data_sample = np.delete(data, -1, axis=1)
    data_end_colume=data.T[-1]
    data_end_colume=np.reshape(data_end_colume,(-1,1))
    data_label=one_hot_transform(data_end_colume,data_end_colume)
    return data_sample,data_label,data_end_colume

def get_random_block_form_data(data_sampls,data_labels,data_cloumn,batch_size):
    start_index=np.random.randint(0,len(data_sampls)-batch_size)
    return data_sampls[start_index:(start_index+batch_size)],data_labels[start_index:(start_index+batch_size)],data_cloumn[start_index:(start_index+batch_size)]

def train(data_samples,data_labels,data_tag):
    # 定义输出为4维矩阵的placeholder
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,
        Dscdc_inference.IMAGE_WIDTH,
        Dscdc_inference.IMAGE_CHANNELS],
                       name='x-input')

    y_ = tf.placeholder(tf.float32, [None, Dscdc_inference.IMAGE_LABELS], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = Dscdc_inference.inference(x, False, regularizer)

    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # cross_entropy = -tf.reduce_sum(y_ * tf.log(y+1e-10))
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        data_samples.shape[0] / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')


    # 初始化TensorFlow持久化类。
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        tf.global_variables_initializer().run()
        time_start=time.time()
        for i in range(TRAINING_STEPS):
            xs, ys,ys_pca = get_random_block_form_data(data_samples,data_labels,data_tag,BATCH_SIZE)

            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                Dscdc_inference.IMAGE_WIDTH,
                Dscdc_inference.IMAGE_CHANNELS))
            _,loss_value, step = sess.run([train_op,loss,global_step], feed_dict={x: reshaped_xs, y_: ys})
            if i % 100 == 0:
                print("After %d training step(s), loss on training batch is %g" % (step, loss_value))
                saver.save(sess, os.path.join(SAVE_PATH, MODEL_NAME), global_step=global_step)
            if i%5000==0:
                time_end=time.time()
                print("time uses %f"%(time_end-time_start))


def main(argv=None):
    train_data=matfile_reader.dataset_reader(r'G:\04 实验数据\bearing_dataset_2000.mat')
    print("train dateset read over")
    train_samples, train_labels,data_tag= data_preprocess(train_data)
    train(train_samples,train_labels,data_tag)

if __name__ == '__main__':
    main()