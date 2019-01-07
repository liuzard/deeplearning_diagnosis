import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from conv_dae_bearing import matfile_reader
import sklearn.preprocessing as prep

DATA_PATH='E:\\bearing_dataset_1024.mat'

#bearing dataset reader
dataset=matfile_reader.dataset_reader(DATA_PATH)
max,min=np.max(dataset),np.min(dataset)
print(max,min)
# preprocessor=prep.StandardScaler().fit(dataset)
# dataset=preprocessor.transform(dataset)

#get data(size=batch_size) randomly from dataset
def get_random_block_form_data(data_sampls,batch_size):
    start_index=np.random.randint(0,len(data_sampls)-batch_size)
    return data_sampls[start_index:(start_index+batch_size)]



# #mnist reader
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('C:\\Users\liuzard\PycharmProjects\deep_learning_diagnosis\MNIST_data', validation_size=0, one_hot=False)
#
# img = mnist.train.images[21]
# plt.imshow(img.reshape((28, 28)))


#input
inputs_ = tf.placeholder(tf.float32, (None, 32, 32, 1), name='inputs_')
targets_ = tf.placeholder(tf.float32, (None, 32, 32, 1), name='targets_')

#encoder
#conv1:32x32 to 16x16
conv1 = tf.layers.conv2d(inputs_, 64,3,padding='same', activation=tf.nn.relu)
print(conv1)
conv1 = tf.layers.max_pooling2d(conv1, 2, 2, padding='same')
print(conv1)

#conv2:16x16 to 8x8
conv2 = tf.layers.conv2d(conv1, 64, (3,3), padding='same', activation=tf.nn.relu)
conv2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')


conv3 = tf.layers.conv2d(conv2, 32, (3,3), padding='same', activation=tf.nn.relu)
conv3 = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')
print(conv3)

#decoder
conv4 = tf.image.resize_nearest_neighbor(conv3, (8,8))
conv4 = tf.layers.conv2d(conv4, 32, (3,3), padding='same', activation=tf.nn.relu)

conv5 = tf.image.resize_nearest_neighbor(conv4, (16,16))
conv5 = tf.layers.conv2d(conv5, 64, (3,3), padding='same', activation=tf.nn.relu)

conv6 = tf.image.resize_nearest_neighbor(conv5, (32,32))
conv6 = tf.layers.conv2d(conv6, 64, (3,3), padding='same', activation=tf.nn.relu)

#logits and outputs

logits_ = tf.layers.conv2d(conv6, 1, (3,3), padding='same', activation=None)

outputs_ = tf.nn.sigmoid(logits_, name='outputs_')

#loss and optimizer1

loss = tf.reduce_mean(tf.pow(tf.subtract(outputs_,targets_),2))
cost = tf.reduce_mean(loss)
#loss and optimizer2
# loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_,logits=logits_)
# cost=tf.reduce_mean(loss)

optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)

#train
sess = tf.Session()
noise_factor = 0.5
epochs = 5
batch_size = 128
sess.run(tf.global_variables_initializer())

for e in range(epochs):
    for idx in range(dataset.shape[0]// batch_size):
        batch = get_random_block_form_data(dataset,batch_size)
        imgs = batch.reshape((-1, 32, 32, 1))

        # 加入噪声
        noisy_imgs=imgs
        # noisy_imgs = imgs + noise_factor * np.random.randn(*imgs.shape)
        # noisy_imgs = np.clip(noisy_imgs, 0., 1.)
        batch_cost, _ = sess.run([cost, optimizer],
                                 feed_dict={inputs_: noisy_imgs,
                                            targets_: imgs})

        print("Epoch: {}/{} ".format(e + 1, epochs),
              "Training loss: {:.4f}".format(batch_cost))

#plot
# fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
# in_imgs = mnist.test.images[10:20]
# noisy_imgs = in_imgs + noise_factor * np.random.randn(*in_imgs.shape)
# noisy_imgs = np.clip(noisy_imgs, 0., 1.)
#
# reconstructed = sess.run(outputs_,
#                          feed_dict={inputs_: noisy_imgs.reshape((10, 28, 28, 1))})
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