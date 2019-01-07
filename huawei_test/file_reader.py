import numpy as np
file_name='C:\\Users\liuzard\PycharmProjects\deep_learning\huawei_test\用例示例\TrainData_2015.1.1_2015.2.19.txt'

def file_trans(file):
    data_list=[]
    date_list=[]
    with open(file) as file_object:
        for index,item in file_object:
            print(index)
    #         line_all=line.split()
    #         line_date=line_all[2][:10]
    #         line_data=int(line_all[1][6:])
    #         line_data_and_date=[line_date,line_data]
    #         data_list.append(line_data_and_date)
    #         date_list.append(line_date)
    #     date_list=sorted(set(date_list),key=date_list.index)
    # print(data_list)


    # print(data_list)
    # print(data_list)
    # data_matrix=np.zeros([len(date_list),16])
    # data_matrix[:,0]=date_list
    # for i in range(len(data_list)):
    #     for j in range(data_matrix.shape[0]):
    #         if int(data_list[i][0])==int(data_matrix[j][0]):
    #             if int(data_list[i][1])<16:
    #                 data_matrix[j][int(data_list[i][1])]=data_matrix[j][int(data_list[i][1])]+1
    # print(data_matrix)
    # return data_matrix


train_data=file_trans(file_name)
