# coding=utf-8
import datetime
import operator
import math
import tensorflow as tf
from huawei_test import file_reader_2
import numpy as np
from itertools import groupby



def predict_vm(ecs_lines, input_lines,test_lines):
    # Do your work from here#
    result = []
    if ecs_lines is None:
        print ('ecs information is none')
        return result
    if input_lines is None:
        print ('input file information is none')
        return result

    # transform the trianing data and testdata to matrix
    data_tran = trans_data(ecs_lines)
    data_tran_test = trans_data(test_lines)

    # tansform the input txt to the wanted condition
    input_condition = []
    for line in input_lines:
        sub_list = line.split(' ')
        input_condition.append(sub_list)

    # the configration of physical server
    server_info = input_condition[0]
    server_cpu = int(server_info[0])
    server_mem = int(server_info[1])
    # the flavor to be predicted
    input_flavor_num = int(input_condition[1][0])
    flavor_info = input_condition[2:2+input_flavor_num]
    flavor_name = [flavor[0]for flavor in flavor_info]
    flavor_name_int = [int(flavor[6:]) for flavor in flavor_name ]# need a sort
    cpu_use_list = [int(flavor[1]) for flavor in flavor_info]
    mem_use_list = [int(int(flavor[2])/1024) for flavor in flavor_info]
    flavor_list = []
    for i in range(len(flavor_name)):
        flavor_list.append([flavor_name[i],cpu_use_list[i],mem_use_list[i]])
    print (flavor_list)

    # the optimiztion dim
    optimize_dim = input_condition[2+input_flavor_num][0]

    # prognosis start and end
    prog_start = datetime.datetime.strptime(input_condition[3 + input_flavor_num][0], '%Y-%m-%d')
    prog_end = datetime.datetime.strptime(input_condition[4 + input_flavor_num][0], '%Y-%m-%d')
    day_dur = (prog_end - prog_start).days + 1

    # predict stage--Armodel
    prog_used_num = 3
    batch_size = 10
    train_step = 1000
    learning_rate = 0.001

    dates = [date[0] for date in data_tran]
    weeks = []
    for date in dates:
        dl = datetime.datetime.strptime(date, '%Y-%m-%d')
        date_tuple = dl.isocalendar()
        week = '第' + str(date_tuple[1]) + '周'
        weeks.append(week)

    flavor_pre_used = []
    for i in flavor_name_int:
        flavor=[flavor[i] for flavor in data_tran]
        xy_map = []
        for x, y in groupby(zip(weeks, flavor), key=lambda _: _[0]):
            y_list = [v for _, v in y]
            xy_map.append([x, sum(y_list) / len(y_list)])
        print(xy_map)
        x_unique, y_mean = [*zip(*xy_map)]

        train_X, train_y = generate_data(y_mean,prog_used_num)
        train_X = np.squeeze(train_X)

        x = tf.placeholder(dtype=tf.float32, shape=[None, prog_used_num], name="x-input")
        y_target = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y_target')

        weights = tf.Variable(tf.truncated_normal([prog_used_num, 1], stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, shape=[1]))
        y_out = tf.matmul(x, weights) + bias

        loss = 0.5 * tf.reduce_mean(tf.square(y_out - y_target))
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        use_matrix = train_X[-1]
        use_matrix = use_matrix.reshape(1, -1)

        with tf.Session() as  sess:
            sess.run(tf.global_variables_initializer())
            for i in range(train_step):
                x_in, y_in = get_random_block_form_data(train_X, train_y, batch_size=batch_size)
                # print(x_in,y_in)
                sess.run(train_op, feed_dict={x: x_in, y_target: y_in})
                # if i % 10 == 0:
                #     loss_value = sess.run(loss, feed_dict={x: train_X, y_target: train_y})
                #     print(loss_value)

            print(sess.run(weights))
            pre_value = sess.run(7 * y_out, feed_dict={x: use_matrix})
            flavor_pre_used.append(int(pre_value))
    print(flavor_pre_used)
    flavor_used_test = sum_of_flavor(data_tran_test, flavor_name_int)

    predict_score=test_pre_acc(flavor_used_test,flavor_pre_used)
    print('预测得分：',predict_score*100)

    # put the flavors into physical server
    pre_used_sum = sum(flavor_pre_used)  # sum of predicted flavor
    result.append(pre_used_sum)
    for i in range(len(flavor_pre_used)):
        flavor_str = flavor_name[i] + ' ' + str(flavor_pre_used[i])
        result.append(flavor_str)
    result.append('')

    for i in range(len(flavor_list)):
        flavor_list[i].append(flavor_pre_used[i])
    # print flavor_list
    if optimize_dim == 'CPU':
        flavor_list_sort = sorted(flavor_list,key=operator.itemgetter(1,2))
    else:
        flavor_list_sort = sorted(flavor_list,key=operator.itemgetter(2,1))
    flavor_pre_used_sort=[flavor[3] for flavor in flavor_list_sort]
    server_list = []
    while sum(flavor_pre_used_sort) > 0:
        mem_remain = server_mem
        cpu_remain = server_cpu
        server_flavor = []
        i=len(flavor_pre_used_sort)-1
        while i >= 0:
            if cpu_remain>=flavor_list[i][1] and mem_remain >= flavor_list[i][2]:
                n_cpu = int(cpu_remain/flavor_list[i][1])
                n_mem = int(mem_remain/flavor_list[i][2])
                n = min(n_cpu,n_mem)
                if n >= flavor_pre_used_sort[i]:
                    n = flavor_pre_used_sort[i]
                cpu_remain = cpu_remain-n*flavor_list[i][1]
                mem_remain = mem_remain-n*flavor_list[i][2]
                flavor_pre_used_sort[i] = flavor_pre_used_sort[i]-n
                server_flavor.insert(0,n)
            else:
                server_flavor.insert(0,0)
            i=i-1
        server_list.append(server_flavor)
    sever_num=len(server_list)
    result.append(sever_num)
    for server_index in range(sever_num):
        sever=server_list[server_index]
        server_str=str(server_index+1)
        for flavor_index in range(len(sever)):
            if sever[flavor_index] !=0:
               server_str=server_str + ' ' + flavor_list[flavor_index][0] + ' '+ str(sever[flavor_index])
        result.append(server_str)

    # compute the predict score
    # predict_score =

    # compute the palce score
    server_config=[server_cpu,server_mem]
    place_score=place_envaluate(server_config,optimize_dim,sever_num,flavor_list)
    print('资源利用率:',place_score*100,'%')

    return result


def trans_data(test_lines):
    date_list = []
    data_list = []
    for record in test_lines:
        record_seperate = record.split()
        record_date = record_seperate[2][:10]
        record_value = int(record_seperate[1][6:])
        line_data_and_value = [record_date, record_value]
        data_list.append(line_data_and_value)
        date_list.append(record_date)
    date_list = sorted(set(date_list), key=date_list.index)
    data_tran = []
    for date in date_list:
        sub_date = [date]
        for i in range(15):  # struct the matrix,the matrix have 16 column
            sub_date.append(0)
        data_tran.append(sub_date)

    curren_index = 0  # index of date,if date changes,index plus 1
    for i in range(len(data_list)):
        if i == 0:
            data_tran[curren_index][data_list[i][1]] = data_tran[curren_index][data_list[i][1]] + 1
        else:
            if data_list[i][0] == data_list[i - 1][0]:
                if data_list[i][1] < 16:
                    data_tran[curren_index][data_list[i][1]] = data_tran[curren_index][data_list[i][1]] + 1
                else:
                    pass
            else:
                curren_index = curren_index + 1
                if data_list[i][1] < 16:
                    data_tran[curren_index][data_list[i][1]] = data_tran[curren_index][data_list[i][1]] + 1
                else:
                    pass
    print(curren_index)
    return data_tran

def sum_of_flavor(flavor_data,flavor_name,rate = 1.0):
    flavor_pre_sum = []
    for i in range(1, len(flavor_data[0])):
        item_mean = sum([data[i] for data in flavor_data])
        item_mean = int(rate*item_mean)
        # item_mean=int(1.26*item_mean)
        flavor_pre_sum.append(item_mean)
    flavor_pre_used = [flavor_pre_sum[index - 1] for index in flavor_name]
    return flavor_pre_used


def test_pre_acc(real_data,predict_data):
    flavor_num = len(predict_data)
    fenmu1 = math.sqrt(sum([data*data for data in predict_data])/flavor_num)
    fenmu2 = math.sqrt(sum([data*data for data in real_data])/flavor_num)
    fenzi=0
    for i in range(flavor_num):
        fenzi=fenzi+(predict_data[i]-real_data[i])**2
    fenzi=math.sqrt(fenzi/flavor_num)
    predict_score=1-(fenzi/(fenmu1+fenmu2))
    return predict_score

def place_envaluate(server_config,optim_dim,sever_num,flavor_list):
    if optim_dim == 'cpu':
        cpu_use = sum([flavor[1]*flavor[3] for flavor in flavor_list])
        utilization = cpu_use/(sever_num*server_config[0])
    else:
        mem_use = sum([flavor[2]*flavor[3] for flavor in flavor_list])
        utilization=mem_use/(sever_num*server_config[1])
    return utilization

def generate_data(seq,prog_used_num):
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