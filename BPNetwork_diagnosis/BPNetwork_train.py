# -*-coding utf-8 -*-
#--------CHAPTER05 MNIST最佳样例实践之二：训练模块------------------#
import os
import numpy as np
import pandas as pd
import sklearn.preprocessing as pre
import tensorflow as tf
from BPNetwork_diagnosis import matfile_reader
import time

from BPNetwork_diagnosis import BPNetwork_inference

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

BATCH_SIZE=100
BAES_LEARNING_RATE=0.8
LEARNING_RATE_DECAY=0.99
AVERAGE_MOVING_DECAY=0.99
REGULARZITION_RATE=0.0001
TRAINING_STEPS=10000

MODEL_SAVE_PATH=r"G:\06 深度学习模型\BPNetwork\BPNetwork_diagnosis"
MODEL_NAME="BPNetwork_diagnosis.ckpt"

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
    x=tf.placeholder(dtype=tf.float32, shape=[None, BPNetwork_inference.INPUT_NODE], name="input")
    y_=tf.placeholder(dtype=tf.float32, shape=[None, BPNetwork_inference.OUTPUT_NODE], name="labels")
    regularizer=tf.contrib.layers.l2_regularizer(REGULARZITION_RATE)
    y= BPNetwork_inference.inference(x, regularizer)
    global_step=tf.Variable(0,trainable=False)

    variable_average=tf.train.ExponentialMovingAverage(AVERAGE_MOVING_DECAY,global_step)
    average_op=variable_average.apply(tf.trainable_variables())
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
    learing_rate=tf.train.exponential_decay(BAES_LEARNING_RATE,global_step,data_samples.shape[0] / BATCH_SIZE,LEARNING_RATE_DECAY,staircase=True)

    train_step=tf.train.GradientDescentOptimizer(learing_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,average_op]):
        # train_op=tf.no_op(name='train')
        train_op = tf.no_op()

    saver=tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        time_start=time.time()
        for i in range(TRAINING_STEPS):
            xs, ys, ys_pca = get_random_block_form_data(data_samples, data_labels, data_tag, BATCH_SIZE)

            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            if i%100==0:
                print("After %d training steps,the loss of the model is %g"%(step,loss_value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
            if i%5000==0:
                time_end=time.time()
                print("time use %f"%(time_end-time_start))


def main(argv=None):
    # mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    train_data=matfile_reader.dataset_reader(r'G:\04 实验数据\bearing_dataset_2000.mat')
    # train_data = pd.read_excel("traindataset.xlsx")
    print("train dateset read over")
    # test_data = pd.read_excel("E:\\故障诊断实验数据\\实验数据_20171201\\原始数据_数据集\\test_dataset.xlsx")

    train_samples, train_labels,data_tag= data_preprocess(train_data)
    train(train_samples,train_labels,data_tag)

if __name__ == '__main__':
    main()