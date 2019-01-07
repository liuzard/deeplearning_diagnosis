import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from conv_dae_bearing import matfile_reader

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', validation_size=0, one_hot=False)

DATA_PATH='E:\\bearing_dataset_1024.mat'

#bearing dataset reader
dataset=matfile_reader.dataset_reader(DATA_PATH)
def get_random_block_form_data(data_sampls,batch_size):
    start_index=np.random.randint(0,len(data_sampls)-batch_size)
    return data_sampls[start_index:(start_index+batch_size)]



hidden_units = 200

image_size = dataset.shape[1]

# Input
inputs_ = tf.placeholder(tf.float32, [None, image_size], name='inputs_')
targets_ = tf.placeholder(tf.float32, [None, image_size], name='targets_')

# hidden
hidden_layer = tf.layers.dense(inputs_, hidden_units, activation=tf.nn.relu)

# logits和outputs
logits_ = tf.layers.dense(hidden_layer, image_size, activation=None)

outputs_ = tf.sigmoid(logits_, name='outputs_')

# loss
cost= tf.reduce_sum(tf.pow(tf.subtract(outputs_,targets_),2))
# cost = tf.reduce_mean(loss)

# optimizer
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

sess = tf.Session()

noise_factor = 0
epochs = 10
batch_size = 128
sess.run(tf.global_variables_initializer())
for e in range(epochs):
    for ii in range(mnist.train.num_examples//batch_size):
        imgs = get_random_block_form_data(dataset,batch_size)

        # 加入噪声
        noisy_imgs = imgs + noise_factor * np.random.randn(*imgs.shape)
        # noisy_imgs = np.clip(noisy_imgs, 0., 1.)

        batch_cost, _ = sess.run([cost, optimizer],
                                 feed_dict={inputs_: noisy_imgs,
                                            targets_: imgs})

        print("Epoch: {}/{}...".format(e+1, epochs),
              "Training loss: {:.4f}".format(batch_cost))
#
# fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
# imgs = mnist.test.images[10:20]
# # 加入噪声
# noisy_imgs = imgs + noise_factor * np.random.randn(*imgs.shape)
# noisy_imgs = np.clip(noisy_imgs, 0.0, 1.0)
#
# reconstructed = sess.run(outputs_, feed_dict={inputs_: imgs})
#
# for images, row in zip([noisy_imgs, reconstructed], axes):
#     for img, ax in zip(images, row):
#         ax.imshow(img.reshape((28, 28)))
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#
# fig.tight_layout(pad=0.1)
# plt.show()
sess.close()