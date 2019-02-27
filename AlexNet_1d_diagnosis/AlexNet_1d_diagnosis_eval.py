import os
import time

import numpy as np
import tensorflow as tf
from sklearn import decomposition

from AlexNet_1d_diagnosis import AlexNet_1d_diagnosis_inference
from AlexNet_1d_diagnosis import AlexNet_1d_diagnosis_train
from AlexNet_1d_diagnosis import matfile_reader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
EVAL_INTERVAL_SEC = 3
num_test = 684
batch_size = 2000


def pca_analysis(data_sampel, num_components):
    pca = decomposition.PCA(n_components=num_components)
    pca_X = pca.fit_transform(data_sampel)
    return pca_X


def get_random_block_form_data(data_sampls, data_labels, batch_size):
    start_index = np.random.randint(0, len(data_sampls) - batch_size)
    return data_sampls[start_index:(start_index + batch_size)], data_labels[start_index:(start_index + batch_size)]


def evaluate(data_samples, data_labels, datatag):
    with tf.Graph().as_default() as g:
        # num_test=mnist.test.num_examples
        x = tf.placeholder(tf.float32, [
            batch_size,
            AlexNet_1d_diagnosis_inference.IMAGE_WIDTH,
            AlexNet_1d_diagnosis_inference.IMAGE_CHANNELS],
                           name='x-input')
        y_ = tf.placeholder(dtype=tf.float32, shape=(None, AlexNet_1d_diagnosis_inference.IMAGE_LABELS), name="y-input")
        hidden, y = AlexNet_1d_diagnosis_inference.inference(x, train=False, regularizer=None)
        # validation_feed={x:mnist.validation.images,y_:mnist.validation.labels}
        correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predict, dtype=tf.float32))
        variable_averages = tf.train.ExponentialMovingAverage(AlexNet_1d_diagnosis_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        while True:
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                xs, ys = get_random_block_form_data(data_samples, data_labels, batch_size)
                reshaped_xs = np.reshape(xs, (
                    batch_size,
                    AlexNet_1d_diagnosis_inference.IMAGE_WIDTH,
                    AlexNet_1d_diagnosis_inference.IMAGE_CHANNELS))
                ckpt = tf.train.get_checkpoint_state(AlexNet_1d_diagnosis_train.SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    time_start = time.time()
                    accuracy_value, hidden_value = sess.run([accuracy, hidden], feed_dict={x: reshaped_xs, y_: ys})
                    print("After %s trainning steps ,the accuracy on test datasets of model is %f" % (
                    global_step, accuracy_value))
                    time_end = time.time()
                    print(time_end - time_start)
                    #
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
    test_data = matfile_reader.dataset_reader(r'G:\04 实验数据\02 实验台实验数据\实验数据_20180119\bearing_dataset_2000.mat', train=False)
    # test_data=test_data[0:10000,:]
    print("train dateset read over")

    # test_data = pd.read_excel("E:\\故障诊断实验数据\\实验数据_20171201\\原始数据_数据集\\test_dataset.xlsx")

    test_samples, test_labels, data_tag = AlexNet_1d_diagnosis_train.data_preprocess(test_data)

    evaluate(test_samples, test_labels, data_tag)


if __name__ == "__main__":
    tf.app.run()
