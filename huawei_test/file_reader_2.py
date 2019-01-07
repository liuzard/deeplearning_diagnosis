
flavor_num=15#读取服务器规格的种类

def file_trans(file):
    data_list=[]
    date_list=[]
    with open(file) as file_object:
        for line in file_object:
            line_all=line.split()
            line_date=line_all[2][:10]
            line_value=int(line_all[1][6:])
            line_data_and_value=[line_date,line_value]
            data_list.append(line_data_and_value)
            date_list.append(line_date)
        date_list=sorted(set(date_list),key=date_list.index)

    #构造转换列表格式，每行16列，第一列为日期，另外15列为该日期15种规格服务器使用情况
    data_tran=[]
    for date in date_list:
        sub_date=[date]
        for i in range(flavor_num):#给每一个日期添加15个数字，初始化为0，代表该天试用flavorx的数量
            sub_date.append(0)
        data_tran.append(sub_date)

    curren_index=0#指示当前处理到哪个日期，当数据日期发生变化时+1
    for i in range(len(data_list)):
        if i==0:
            data_tran[curren_index][data_list[i][1]]=data_tran[curren_index][data_list[i][1]]+1
        else:
            if data_list[i][0]==data_list[i-1][0]:
                if data_list[i][1]<=flavor_num:
                    data_tran[curren_index][data_list[i][1]] = data_tran[curren_index][data_list[i][1]] + 1
                else:
                    pass
            else:
                curren_index = curren_index + 1
                if data_list[i][1]<=flavor_num:
                    data_tran[curren_index][data_list[i][1]] = data_tran[curren_index][data_list[i][1]] + 1
                else:
                    pass
    return data_tran

data=file_trans("C:\\Users\liuzard\PycharmProjects\deep_learning\huawei_test\练习数据\data_2015_1-5.txt")
