
# ---------导入数学运算模块、数据预处理模块、tensorflow,mnist数据集---------------#
import numpy as np
import tensorflow as tf
import sklearn.preprocessing as pre
import sklearn.preprocessing as prep
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('/tmp/MNIST_data',one_hot=True)
sess=tf.InteractiveSession()

#-------------------------导入数据，并预处理--------#
def one_hot_transform(formalarray,input_label):#把原始编码转换成One-hot编码
    enc=pre.OneHotEncoder()
    enc.fit(formalarray)
    return enc.transform(input_label).toarray()

def data_preprocess(data):#对原始数据进行处理，得到样本和样本标签
    data=np.array(data)
    data_sample=np.delete(data,-1,axis=1)
    data_end_colume=data.T[-1]
    data_end_colume=np.reshape(data_end_colume,(-1,1))
    data_label=one_hot_transform(data_end_colume,data_end_colume)
    return data_sample,data_label

# --------自编码器权重初始化------------#

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6 / (fan_in + fan_out))
    high = constant * np.sqrt(6 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

# --------高斯去噪自编码器类定义------------#
class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.relu, optimizer=tf.train.AdamOptimizer(), scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x+scale*tf.random_normal((n_input,)), self.weights['w1']), self.weights['b1']))
        # self.hidden = self.transfer(tf.add(tf.matmul(self.x , self.weights['w1']), self.weights['b1']))
        self.reconstruction=tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])
        self.cost = 1*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,self.x),2.0))+\
        +0*tf.reduce_sum(tf.pow(tf.subtract(tf.matmul(self.reconstruction,tf.transpose(self.reconstruction)),tf.matmul(self.x,tf.transpose(self.x))),2.0))
        self.optimizer=optimizer.minimize(self.cost)
        init=tf.global_variables_initializer()
        self.sess=tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights=dict()
        # all_weights['w1']=tf.Variable(xavier_init(self.n_input,self.n_hidden))
        # all_weights['b1']=tf.Variable(tf.zeros([self.n_hidden],dtype=tf.float32))
        # all_weights['w2']=tf.Variable(xavier_init(self.n_hidden,self.n_input))
        # all_weights['b2']=tf.Variable(tf.zeros([self.n_input],dtype=tf.float32))
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(xavier_init(self.n_hidden,self.n_input))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        cost,opt=self.sess.run((self.cost,self.optimizer),feed_dict={self.x: X,self.scale:self.training_scale})
        return cost

    def cac_total_cost(self, X):
        return self.sess.run(self.cost,feed_dict={self.x: X,self.scale:self.training_scale})

    def transform(self,X):
        return self.sess.run(self.hidden,feed_dict={self.x: X, self.scale:self.training_scale})

    def generate(self,hidden=None):
        if hidden is None:
            hidden=np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction,feed_dict={self.hidden:hidden})

    def reconstruct (self,X):
        return self.sess.run(self.reconstruction,feed_dict={self.x:X,self.scale:self.training_scale})

    def getweights(self):
        return self.sess.run(self.weights['w1'])


    def getBiases(self):
        return self.sess.run(self.weights['b1'])
#-------------------------数据标准化处理，让数据变成0均值，标准差为1的分布-------------
def standard_scale(X_train,X_test):
    preprocessor=prep.StandardScaler().fit(X_train)
    x_train=preprocessor.transform(X_train)
    x_test=preprocessor.transform(X_test)
    return x_train,x_test

#---------------从数据集里面随机取数据---------------------
def get_random_block_form_data(data_sampls,data_labels,batch_size):
    start_index=np.random.randint(0,len(data_sampls)-batch_size)
    return data_sampls[start_index:(start_index+batch_size)],data_labels[start_index:(start_index+batch_size)]

# #
# # train_data=pd.read_excel("traindataset.xlsx")
# # test_data=pd.read_excel("testdataset.xlsx")
#
# traimnist.train.num_examples,train_labels=data_preprocess(train_data)
# test_samples,test_labels=data_preprocess(test_data)
# train_data_processed=np.column_stack((traimnist.train.num_examples,train_labels))

# traimnist.train.num_examples,test_samples=standard_scale(traimnist.train.num_examples,test_samples)

# mnist.train.num_examples=traimnist.train.num_examples.shape[0]
training_epochs=0
batch_size=100
display_step=1


autoencoder1=AdditiveGaussianNoiseAutoencoder(n_input=784,n_hidden=600,transfer_function=tf.nn.relu, optimizer=tf.train.AdamOptimizer(learning_rate=0.001),scale=0.01)

for epoch in range(training_epochs):
    avg_cost=0
    total_batch=int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
        batch_xs,batch_ys=mnist.validation.next_batch(batch_size)
        cost=autoencoder1.partial_fit(batch_xs)
        avg_cost+=cost/mnist.train.num_examples*batch_size
    if epoch % display_step==0:
        print("Epoch:",'%04d'%(epoch+1),"cost=","{:.5f}".format(avg_cost))

# print("Total cost:"+str(autoencoder1.cac_total_cost(test_samples)))
# extract_features=autoencoder1.transform(test_samples)
# print("extacted feature:"+str(extract_features))


autoencoder2=AdditiveGaussianNoiseAutoencoder(autoencoder1.n_hidden,n_hidden=300,transfer_function=tf.nn.relu, optimizer=tf.train.AdamOptimizer(learning_rate=0.001),scale=0.01)
h1_train_features=autoencoder1.transform(mnist.train.images)
# h1_test_features=autoencoder1.transform(test_samples)
for epoch in range(training_epochs):
    avg_cost=0
    total_batch=int(mnist.train.num_examples/batch_size)
    for i in range(10*total_batch):
        batch_xs, batch_ys=get_random_block_form_data(h1_train_features,h1_train_features,batch_size)
        cost=autoencoder2.partial_fit(batch_xs)
        avg_cost+=cost/mnist.train.num_examples*batch_size
   # if epoch % display_step==0:
    print("Epoch:",'%04d'%(epoch+1),"cost=","{:.5f}".format(avg_cost))

# print("Total cost:"+str(autoencoder2.cac_total_cost(h1_test_features)))
# print("extacted feature:"+str(autoencoder2.transform(h1_test_features)))

autoencoder3=AdditiveGaussianNoiseAutoencoder(autoencoder2.n_hidden,n_hidden=200,transfer_function=tf.nn.relu, optimizer=tf.train.AdamOptimizer(learning_rate=0.001),scale=0.01)
h2_train_features=autoencoder2.transform(h1_train_features)
# h2_test_features=autoencoder2.transform(h1_test_features)
for epoch in range(training_epochs):
    avg_cost=0
    total_batch=int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys=get_random_block_form_data(h2_train_features,h2_train_features,batch_size)
        cost=autoencoder3.partial_fit(batch_xs)
        avg_cost+=cost/mnist.train.num_examples*batch_size
   # if epoch % display_step==0:
    print("Epoch:",'%04d'%(epoch+1),"cost=","{:.5f}".format(avg_cost))

# print("Total cost:"+str(autoencoder3.cac_total_cost(h2_test_features)))
# print("extacted feature:"+str(autoencoder3.transform(h2_test_features)))


#---------------------------------------第一层自编码器参数----------------------------------------------------------------
#W3=tf.Variable(tf.truncated_normal([autoencoder.n_input,autoencoder.n_hidden],stddev=0.1))
b3=tf.Variable(autoencoder1.getBiases(),trainable=True)
W3=tf.Variable(autoencoder1.getweights(),trainable=True)
#W3=sess.run(w3)
#-----------------------------------------------第二层自编码器参数----------------------------------------------------------
b4=tf.Variable(autoencoder2.getBiases(),trainable=True)
W4=tf.Variable(autoencoder2.getweights(),trainable=True)

#------------------------------------第三层自编码器参数---------------------------------------------------------------------
b5=tf.Variable(autoencoder3.getBiases(),trainable=True)
W5=tf.Variable(autoencoder3.getweights(),trainable=True)

hidden=100
W6=tf.Variable(tf.random_normal([autoencoder3.n_hidden,hidden]))
b6=tf.Variable(tf.zeros([hidden]))


W7=tf.Variable(tf.random_normal([hidden,10]))
b7=tf.Variable(tf.zeros([10]))
x=tf.placeholder(tf.float32,[None,autoencoder1.n_input])

keep_prob=tf.placeholder(tf.float32)

hidden1=tf.nn.relu(tf.matmul(x,W3)+b3)
hidden1_drop=tf.nn.dropout(hidden1,keep_prob)

hidden2=tf.nn.relu(tf.matmul(hidden1_drop,W4)+b4)
hidden2_drop=tf.nn.dropout(hidden2,keep_prob)

hidden3=tf.nn.relu(tf.matmul(hidden2_drop,W5)+b5)
hidden3_drop=tf.nn.dropout(hidden3,keep_prob)

hidden4=tf.nn.relu(tf.matmul(hidden3_drop,W6)+b6)
hidden4_drop=tf.nn.dropout(hidden4,keep_prob)

y=tf.nn.softmax(tf.matmul(hidden4_drop,W7)+b7)

y_=tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
train_step=tf.train.AdagradOptimizer(0.005).minimize(cross_entropy)

tf.global_variables_initializer().run()

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
for j in range (30000):
    batch_xs,batch_ys = get_random_block_form_data(mnist.train.images,mnist.train.labels,100)
    train_step.run({x:batch_xs,y_:batch_ys,keep_prob:1})
    if j%100==0:
        print(accuracy.eval({x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0}))


writer=tf.summary.FileWriter("G://logs",tf.get_default_graph())
writer.close()

