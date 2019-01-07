#-*- coding: utf-8 -*-
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

from BPNetwork_MNIST import BPNetwork_MNIST_inference
from  BPNetwork_MNIST import BPNetwork_MNIST_train


EVAL_INTERVAL_SECS=5

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x=tf.placeholder(dtype=tf.float32, shape=[None, BPNetwork_MNIST_inference.INPUT_NODE], name="input")
        y_=tf.placeholder(tf.float32, [None, BPNetwork_MNIST_inference.OUTPUT_NODE], name="labels")
        # n_feed={x:mnist.validation.images,y_:mnist.validation.labels}
        y= BPNetwork_MNIST_inference.inference(x, None)
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        variable_averages=tf.train.ExponentialMovingAverage(BPNetwork_MNIST_train.AVERAGE_MOVING_DECAY)
        variables_to_restore=variable_averages.variables_to_restore()
        saver=tf.train.Saver(variables_to_restore)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        while True:
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                xs, ys = mnist.test.images,mnist.test.labels
                ckpt=tf.train.get_checkpoint_state(BPNetwork_MNIST_train.MODEL_SAVE_PATH)
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
    mnist=input_data.read_data_sets('/tmp/data',one_hot=True)
    evaluate(mnist)

if __name__=='__main__':
    tf.app.run()


