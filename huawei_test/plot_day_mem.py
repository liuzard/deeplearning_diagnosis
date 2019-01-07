import pygal
from datetime import datetime
import os
from huawei_test.file_reader_2 import file_trans

test_file_name="C:\\Users\liuzard\PycharmProjects\deep_learning\huawei_test\练习数据\data_2015_1-5.txt"
flavor_info_path="C:\\Users\liuzard\PycharmProjects\deep_learning\huawei_test\\flavor.txt"

def read_lines(file_path):
    if os.path.exists(file_path):
        array = []
        with open(file_path, 'r') as lines:
            for line in lines:
                line = line.rstrip()
                if line != '':
                    sublist=line.split(' ')
                    array.append(sublist)
        return array
    else:
        print ('file not exist: ' + file_path)
        return None


# 读取文件数据，并将数据存储到列表中
data_list=file_trans(test_file_name)

#获取数据列表中的日期
dates=[date[0] for date in data_list]

#获取云服务器配置信息
flavor_info = read_lines(flavor_info_path)
print(flavor_info)


#获取日期所在周数的列表
weeks=[]
for date in dates:
    dl=datetime.strptime(date,'%Y-%m-%d')
    date_tuple=dl.isocalendar()
    weeks.append(date_tuple[1])
print(weeks)

#服务器列表，其中的元素为某规格服务器每天使用的情况
flavor_list=[]
cpu_list=[]
mem_list=[]
for i in range(len(data_list)):
    mem_used = 0
    for j in range(1,len(data_list[0])):
        mem_used = mem_used + data_list[i][j]*(int(flavor_info[j-1][2])/1024)
    mem_list.append(mem_used)
print(len(data_list))


#绘图
line_chart=pygal.Line(x_label_rotation=40,show_minor_x_labels=True)
line_chart.title='各规格服务器日使用情况(用例train_dataset)'
line_chart.x_labels=dates
N=12
line_chart._x_labels_major=dates[::N]
line_name='mem_used'
line_chart.add(line_name,mem_list)
line_chart.render_to_file('flavor_used_mem_month1~5.avg')