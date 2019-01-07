# coding=utf-8
import datetime
import operator
import math
import xlwt


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

    sum1 = 0
    for i in range(len(data_tran)):
        for j in range(1,len(data_tran[0])):
            sum1 += data_tran[i][j]
    print(sum1)




    data_tran_test = trans_data(test_lines)

    # 新建一个excel文件
    file = xlwt.Workbook(encoding='utf-8', style_compression=0)
    # 新建一个sheet
    sheet = file.add_sheet('trian_data')
    for i in range(len(data_tran)):
        for j in range(len(data_tran[0])):
            sheet.write(i, j, data_tran[i][j])
    sheet=file.add_sheet('test_data')
    for i in range(len(data_tran_test)):
        for j in range(len(data_tran_test[0])):
            sheet.write(i, j, data_tran_test[i][j])
    file.save('minni1.xls')




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
    # print flavor_list

    # the optimiztion dim
    optimize_dim = input_condition[2+input_flavor_num][0]

    # prognosis start and end
    prog_start = datetime.datetime.strptime(input_condition[3 + input_flavor_num][0], '%Y-%m-%d')
    prog_end = datetime.datetime.strptime(input_condition[4 + input_flavor_num][0], '%Y-%m-%d')
    day_dur = (prog_end - prog_start).days + 1

    # predict stage
     #
    train_data_used = data_tran[-day_dur:]
    flavor_pre_used = sum_of_flavor(train_data_used,flavor_name_int,2)
    flavor_used_test=sum_of_flavor(data_tran_test,flavor_name_int)

    predict_score=test_pre_acc(flavor_used_test,flavor_pre_used)
    print('预测得分：',predict_score*100)

    # put the flavors into physical server
    pre_used_sum = sum(flavor_pre_used)  # sum of predicted flavor
    result.append(pre_used_sum)
    for i in range(len(flavor_pre_used)):
        flavor_str = flavor_name[i] + ' ' + str(flavor_pre_used[i])
        result.append(flavor_str)
    result.append('')

    # for i in range(len(flavor_list)):
    #     flavor_list[i].append(flavor_pre_used[i])
    # # print flavor_list
    # if optimize_dim == 'CPU':
    #     flavor_list_sort = sorted(flavor_list,key=operator.itemgetter(1,2))
    # else:
    #     flavor_list_sort = sorted(flavor_list,key=operator.itemgetter(2,1))
    # flavor_pre_used_sort=[flavor[3] for flavor in flavor_list]
    server_list = []
    while sum(flavor_pre_used) > 0:
        mem_remain = server_mem
        cpu_remain = server_cpu
        server_flavor = []
        i=len(flavor_pre_used)-1
        while i >= 0:
            if cpu_remain>=flavor_list[i][1] and mem_remain >= flavor_list[i][2]:
                n_cpu = int(cpu_remain/flavor_list[i][1])
                n_mem = int(mem_remain/flavor_list[i][2])
                n = min(n_cpu,n_mem)
                if n >= flavor_pre_used[i]:
                    n = flavor_pre_used[i]
                cpu_remain = cpu_remain-n*flavor_list[i][1]
                mem_remain = mem_remain-n*flavor_list[i][2]
                flavor_pre_used[i] = flavor_pre_used[i]-n
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

