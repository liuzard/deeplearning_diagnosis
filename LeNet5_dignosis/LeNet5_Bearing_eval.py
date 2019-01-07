import time

import numpy as np
import pandas as pd
import tensorflow as tf
from LeNet5_dignosis import LeNet5_Bearing_inference
from LeNet5_dignosis import LeNet5_Bearing_train
from LeNet5_dignosis import matfile_reader

from sklearn import decomposition
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


EVAL_INTERVAL_SEC=10
num_test=684

def pca_analysis(data_sampel,num_components):
    pca=decomposition.PCA(n_components=num_components)
    pca_X=pca.fit_transform(data_sampel)
    return pca_X

def evaluate(data_samples,data_labels,datatag):
    with tf.Graph().as_default() as g:
        # num_test=mnist.test.num_examples
        x = tf.placeholder(tf.float32, [
            data_samples.shape[0],
            LeNet5_Bearing_inference.IMAGE_SIZE,
            LeNet5_Bearing_inference.IMAGE_SIZE,
            LeNet5_Bearing_inference.IMAGE_CHANNELS],
                           name='x-input')
        y_=tf.placeholder(dtype=tf.float32, shape=(None, LeNet5_Bearing_inference.IMAGE_LABELS), name="y-input")
        hidden,y= LeNet5_Bearing_inference.inference(x, train=False, regularizer=None)
        # validation_feed={x:mnist.validation.images,y_:mnist.validation.labels}
        correct_predict=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_predict,dtype=tf.float32))
        variable_averages=tf.train.ExponentialMovingAverage(LeNet5_Bearing_train.MOVING_AVERAGE_DECAY)
        variables_to_restore=variable_averages.variables_to_restore()
        saver=tf.train.Saver(variables_to_restore)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        while True:
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                xs,ys=data_samples,data_labels
                reshaped_xs = np.reshape(xs, (
                    data_samples.shape[0],
                    LeNet5_Bearing_inference.IMAGE_SIZE,
                    LeNet5_Bearing_inference.IMAGE_SIZE,
                    LeNet5_Bearing_inference.IMAGE_CHANNELS))
                ckpt=tf.train.get_checkpoint_state(LeNet5_Bearing_train.SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_value,hidden_value=sess.run([accuracy,hidden],feed_dict={x:reshaped_xs,y_:ys})
                    print("After %s trainning steps ,the accuracy on test datasets of model is %f"%(global_step,accuracy_value))

                    # new_X=pca_analysis(data_samples,3)
                    # fig = plt.figure()
                    # ax = fig.gca(projection='3d')
                    # ax.scatter(new_X[:, 0], new_X[:, 1],new_X[:, 2],c=datatag, cmap=plt.cm.spectral)
                    # plt.show()

                else:
                    print("No checkpoint file found")
                    return
                time.sleep(EVAL_INTERVAL_SEC)
def main(argv=None):
    # test_data = pd.read_excel("testdataset.xlsx")
    test_data=matfile_reader.dataset_reader('E:\\bearing_dataset.mat',train=False)
    print("train dateset read over")

    # test_data = pd.read_excel("E:\\故障诊断实验数据\\实验数据_20171201\\原始数据_数据集\\test_dataset.xlsx")

    test_samples, test_labels,data_tag = LeNet5_Bearing_train.data_preprocess(test_data)

    evaluate(test_samples,test_labels,data_tag)

if __name__=="__main__":
    tf.app.run()