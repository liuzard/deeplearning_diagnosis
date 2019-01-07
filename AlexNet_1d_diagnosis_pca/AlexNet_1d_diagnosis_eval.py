import os
import time

import numpy as np
import tensorflow as tf
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

from AlexNet_1d_diagnosis_pca import AlexNet_1d_diagnosis_inference
from AlexNet_1d_diagnosis_pca import AlexNet_1d_diagnosis_train
from AlexNet_1d_diagnosis_pca import matfile_reader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
EVAL_INTERVAL_SEC = 3
num_test = 684


def tsne_analysis(datasamle, num_components):
    sne = TSNE(n_components=num_components, learning_rate=0.1)
    sne_X = sne.fit_transform(datasamle)
    return sne_X


def pca_analysis(data_sample, num_components):
    pca = decomposition.PCA(n_components=num_components)
    pca_X = pca.fit_transform(data_sample)
    return pca_X


def evaluate(data_samples, data_labels, datatag):
    with tf.Graph().as_default() as g:
        # num_test=mnist.test.num_examples
        x = tf.placeholder(tf.float32, [
            data_samples.shape[0],
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
                xs, ys = data_samples, data_labels
                reshaped_xs = np.reshape(xs, (
                    data_samples.shape[0],
                    AlexNet_1d_diagnosis_inference.IMAGE_WIDTH,
                    AlexNet_1d_diagnosis_inference.IMAGE_CHANNELS))
                ckpt = tf.train.get_checkpoint_state(AlexNet_1d_diagnosis_train.SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_value, hidden_value = sess.run([accuracy, hidden], feed_dict={x: reshaped_xs, y_: ys})

                    print("After %s trainning steps ,the accuracy on test datasets of model is %f" % (
                    global_step, accuracy_value))
                    new_X = pca_analysis(hidden_value, 2)
                    type0_x = []
                    type0_y = []
                    type1_x = []
                    type1_y = []
                    type2_x = []
                    type2_y = []
                    type3_x = []
                    type3_y = []
                    type4_x = []
                    type4_y = []
                    type5_x = []
                    type5_y = []
                    type6_x = []
                    type6_y = []
                    type7_x = []
                    type7_y = []
                    type8_x = []
                    type8_y = []
                    type9_x = []
                    type9_y = []

                    for i in range(2000):
                        if datatag[i, 0] == 0:
                            type0_x.append(new_X[i, 0])
                            type0_y.append(new_X[i, 1])
                        if datatag[i, 0] == 1:
                            type1_x.append(new_X[i, 0])
                            type1_y.append(new_X[i, 1])
                        if datatag[i, 0] == 2:
                            type2_x.append(new_X[i, 0])
                            type2_y.append(new_X[i, 1])
                        if datatag[i, 0] == 3:
                            type3_x.append(new_X[i, 0])
                            type3_y.append(new_X[i, 1])
                        if datatag[i, 0] == 4:
                            type4_x.append(new_X[i, 0])
                            type4_y.append(new_X[i, 1])
                        if datatag[i, 0] == 5:
                            type5_x.append(new_X[i, 0])
                            type5_y.append(new_X[i, 1])
                        if datatag[i, 0] == 6:
                            type6_x.append(new_X[i, 0])
                            type6_y.append(new_X[i, 1])
                        if datatag[i, 0] == 7:
                            type7_x.append(new_X[i, 0])
                            type7_y.append(new_X[i, 1])
                        if datatag[i, 0] == 8:
                            type8_x.append(new_X[i, 0])
                            type8_y.append(new_X[i, 1])
                        if datatag[i, 0] == 9:
                            type9_x.append(new_X[i, 0])
                            type9_y.append(new_X[i, 1])
                    print(len(type8_x), len(type8_y))
                    fig = plt.figure()
                    ax = fig.gca()
                    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
                    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
                    plt.xlabel("PC1", fontsize=20)
                    plt.ylabel("PC2", fontsize=20)
                    plt.xticks(fontsize=20)
                    plt.yticks(fontsize=20)
                    type0 = ax.scatter(type0_x, type0_y, s=20, c='red')
                    type1 = ax.scatter(type1_x, type1_y, s=20, c='green')
                    type2 = ax.scatter(type2_x, type2_y, s=20, c='blue')
                    type3 = ax.scatter(type3_x, type3_y, s=20, c='c')
                    type4 = ax.scatter(type4_x, type4_y, s=20, c='m')
                    type5 = ax.scatter(type5_x, type5_y, s=20, c='y')
                    type6 = ax.scatter(type6_x, type6_y, s=20, c='k')
                    type7 = ax.scatter(type7_x, type7_y, s=20, c='orange')
                    type8 = ax.scatter(type8_x, type8_y, s=20, c='pink')
                    type9 = ax.scatter(type9_x, type9_y, s=20, c='brown')
                    plt.legend((type0, type1, type2, type3, type4, type5, type6, type7, type8, type9),
                               (u'正常', u'内圈（0.18）', u'内圈（0.36）', u'内圈（0.54）',
                                u'外圈（0.18）', u'外圈（0.36）', u'外圈（0.54）', u'滚子（0.18）',
                                u'滚子（0.36）', u'滚子（0.54）'), fontsize=10)

                    plt.show()





                    # ax.scatter(new_X[:, 0], new_X[:, 1],c=datatag[:,0], cmap=plt.cm.spectral,marker='o',linewidths=0.1)
                    # plt.legend((0,1,2), (u'1', u'2', u'3'))


                else:
                    print("No checkpoint file found")
                    return
                time.sleep(EVAL_INTERVAL_SEC)


def main(argv=None):
    # test_data = pd.read_excel("testdataset.xlsx")
    test_data = matfile_reader.dataset_reader('E:\\bearing_dataset_pca.mat', train=False)
    # test_data=test_data[0:10000,:]
    print("train dateset read over")

    # test_data = pd.read_excel("E:\\故障诊断实验数据\\实验数据_20171201\\原始数据_数据集\\test_dataset.xlsx")

    test_samples, test_labels, data_tag = AlexNet_1d_diagnosis_train.data_preprocess(test_data)

    evaluate(test_samples, test_labels, data_tag)


if __name__ == "__main__":
    tf.app.run()
