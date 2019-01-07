import tensorflow as tf
from huawei_test import file_reader_2
import numpy as np
from datetime import datetime
from itertools import groupby

prog_used_num=3
batch_size=10
train_step=1000
learning_rate=0.001


train_file_path="C:\\Users\liuzard\PycharmProjects\deep_learning\huawei_test\official\case\TrainData_2015.1.1_2015.5.24.txt"
train_data=file_reader_2.file_trans(train_file_path)

flavor1= [data[9] for data in train_data] #input_x
print(len(flavor1))

#获取数据列表中的日期
dates=[date[0] for date in train_data]

#获取日期所在周数的列表
weeks=[]
for date in dates:
    dl=datetime.strptime(date,'%Y-%m-%d')
    date_tuple=dl.isocalendar()
    week='第'+str(date_tuple[1])+'周'
    weeks.append(week)
print(len(weeks))

xy_map=[]
for x,y in groupby(zip(weeks,flavor1),key=lambda _:_[0]):
    y_list=[v for _,v in y]
    xy_map.append([x,sum(y_list)/len(y_list)])
print(xy_map)
x_unique,y_mean=[*zip(*xy_map)]



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


train_X,train_y=generate_data(y_mean)
train_X=np.squeeze(train_X)
print(train_X)

print(train_X.shape,train_y.shape)



x=tf.placeholder(dtype=tf.float32,shape=[None,prog_used_num],name="x-input")
y_target=tf.placeholder(dtype=tf.float32,shape=[None,1],name='y_target')


weights=tf.Variable(tf.truncated_normal([prog_used_num,1],stddev=0.1))
bias=tf.Variable(tf.constant(0.1,shape=[1]))
y_out=tf.matmul(x,weights)+bias

loss=0.5*tf.reduce_mean(tf.square(y_out-y_target))
train_op=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


use_matrix=train_X[-1]
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

    print(sess.run(weights))
    pre_matrix = sess.run(7*y_out,feed_dict={x:use_matrix})

    print(pre_matrix)








