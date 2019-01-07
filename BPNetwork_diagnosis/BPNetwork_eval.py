#-*- coding: utf-8 -*-
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

from BPNetwork_diagnosis import BPNetwork_inference
from  BPNetwork_diagnosis import BPNetwork_train
from BPNetwork_diagnosis import matfile_reader

EVAL_INTERVAL_SECS=5

def evaluate(data_samples,data_labels,datatag):
    with tf.Graph().as_default() as g:
        x=tf.placeholder(dtype=tf.float32, shape=[None, BPNetwork_inference.INPUT_NODE], name="input")
        y_=tf.placeholder(tf.float32, [None, BPNetwork_inference.OUTPUT_NODE], name="labels")
        # n_feed={x:mnist.validation.images,y_:mnist.validation.labels}
        y= BPNetwork_inference.inference(x, None)
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        variable_averages=tf.train.ExponentialMovingAverage(BPNetwork_train.AVERAGE_MOVING_DECAY)
        variables_to_restore=variable_averages.variables_to_restore()
        saver=tf.train.Saver(variables_to_restore)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        while True:
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                xs, ys = data_samples, data_labels
                ckpt=tf.train.get_checkpoint_state(BPNetwork_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    global_step=int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
                    accuracy_validation=sess.run(accuracy,feed_dict={x:xs,y_:ys})
                    print("After %d training steps, validation accuracy = %g"%(global_step,accuracy_validation))
                else:
                    print('No checkpoint file found')
                    return
                # writer = tf.summary.FileWriter("G://Logs", tf.get_default_graph())
                # writer.close()
                time.sleep(EVAL_INTERVAL_SECS)



def main(argv=None):
    # test_data = pd.read_excel("testdataset.xlsx")
    test_data = matfile_reader.dataset_reader('E:\\bearing_dataset_2000.mat', train=False)
    print("test dateset read over")

    # test_data = pd.read_excel("E:\\故障诊断实验数据\\实验数据_20171201\\原始数据_数据集\\test_dataset.xlsx")

    test_samples, test_labels, data_tag = BPNetwork_train.data_preprocess(test_data)

    evaluate(test_samples, test_labels, data_tag)

if __name__=='__main__':
    tf.app.run()


