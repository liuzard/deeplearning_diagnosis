import  tensorflow as tf
from huawei_test import file_reader_2
import numpy as np


prog_used_num=7
batch_size=10
train_step=100
learning_rate=0.001


train_file_path="C:\\Users\liuzard\PycharmProjects\deep_learning\huawei_test\official\case\TrainData_2015.5.4_2015.5.24.txt"
test_file_path="C:\\Users\liuzard\PycharmProjects\deep_learning\huawei_test\用例示例\TestData_2015.2.20_2015.2.27.txt"
train_data=file_reader_2.file_trans(train_file_path)
test_data=file_reader_2.file_trans(test_file_path)
flavor1= [data[15] for data in train_data]



def generate_data(seq):
    X = []
    y = []
    # 序列的第i项和后面的prog_used_num-1项合在一起作为输入；第i + prog_used_num项作为输
    # 出。即用sin函数前面的prog_used_num个点的信息，预测第i + prog_used_num个点的函数值。
    for i in range(len(seq) - prog_used_num):
        X.append([seq[i: i + prog_used_num]])
        y.append([seq[i + prog_used_num]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def get_random_block_form_data(data_sampls,data_labels,batch_size):
    start_index=np.random.randint(0,len(data_sampls)-batch_size)
    return data_sampls[start_index:(start_index+batch_size)],data_labels[start_index:(start_index+batch_size)]


train_X,train_y=generate_data(flavor1)
train_X=np.squeeze(train_X)

print(train_X.shape,train_y.shape)



x=tf.placeholder(dtype=tf.float32,shape=[None,prog_used_num],name="x-input")
y_target=tf.placeholder(dtype=tf.float32,shape=[None,1],name='y_target')


weights=tf.Variable(tf.truncated_normal([prog_used_num,1],stddev=0.1))
bias=tf.Variable(tf.constant(0.1,shape=[1]))
y_out=tf.matmul(x,weights)+bias

loss=0.5*tf.reduce_mean(tf.square(y_out-y_target))
train_op=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


use_matrix=train_X[-6:][-1]
use_matrix=use_matrix.reshape(1,-1)

pre_matrix=np.ones([7])
with tf.Session() as  sess:
    sess.run(tf.global_variables_initializer())
    for i in range(train_step):
        x_in,y_in=get_random_block_form_data(train_X,train_y,batch_size=batch_size)
        # print(x_in,y_in)
        sess.run(train_op,feed_dict={x:x_in,y_target:y_in})
        if i%10==0:
            loss_value=sess.run(loss,feed_dict={x:train_X,y_target:train_y})
            print(loss_value)
    for i in range(7):
        if i==0:
            pre_matrix[i]=sess.run(y_out,feed_dict={x:use_matrix})

        else:
            use_real=use_matrix[0][i:].reshape(1,-1)
            use_pre=pre_matrix[0:i].reshape(1,-1)
            use=np.column_stack((use_real,use_pre))
            pre_matrix[i]=sess.run(y_out,feed_dict={x:use})
    print(sum(pre_matrix))






