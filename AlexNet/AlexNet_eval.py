import time


import numpy as np
import tensorflow as tf
from  tensorflow.examples.tutorials.mnist import input_data
from AlexNet import AlexNet_inference
from AlexNet import  AlexNet_train

EVAL_INTERVAL_SEC=10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        num_test=mnist.test.num_examples
        x = tf.placeholder(tf.float32, [
            4000,
            AlexNet_inference.IMAGE_SIZE,
            AlexNet_inference.IMAGE_SIZE,
            AlexNet_inference.IMAGE_CHANNELS],
                           name='x-input')
        y_=tf.placeholder(dtype=tf.float32, shape=(None, AlexNet_inference.IMAGE_LABELS), name="y-input")
        y= AlexNet_inference.inference(x, train=False, regularizer=None)
        validation_feed={x:mnist.validation.images,y_:mnist.validation.labels}
        correct_predict=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_predict,dtype=tf.float32))
        variable_averages=tf.train.ExponentialMovingAverage(AlexNet_train.MOVING_AVERAGE_DECAY)
        variables_to_restore=variable_averages.variables_to_restore()
        saver=tf.train.Saver(variables_to_restore)
        while True:
            with tf.Session() as sess:
                xs,ys=mnist.test.next_batch(4000)
                reshaped_xs = np.reshape(xs, (
                    4000,
                    AlexNet_inference.IMAGE_SIZE,
                    AlexNet_inference.IMAGE_SIZE,
                    AlexNet_inference.IMAGE_CHANNELS))
                ckpt=tf.train.get_checkpoint_state(AlexNet_train.SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_value=sess.run(accuracy,feed_dict={x:reshaped_xs,y_:ys})
                    print("After %s trainning steps ,the accuracy on test datasets of model is %f"%(global_step,accuracy_value))
                else:
                    print("No checkpoint file found")
                    return
                time.sleep(EVAL_INTERVAL_SEC)
def main(argv=None):
    mnist=input_data.read_data_sets("/tmp/data",one_hot=True)
    evaluate(mnist)

if __name__=="__main__":
    tf.app.run()