import pygal
from datetime import datetime
from huawei_test.file_reader_2 import file_trans

test_file_name="C:\\Users\liuzard\PycharmProjects\deep_learning\huawei_test\用例示例\TrainData_2015.1.1_2015.2.19.txt"
#读取文件数据，并将数据存储到列表中
data_list=file_trans(test_file_name)

#获取数据列表中的日期
dates=[date[0] for date in data_list]

#获取日期所在周数的列表
weeks=[]
for date in dates:
    dl=datetime.strptime(date,'%Y-%m-%d')
    date_tuple=dl.isocalendar()
    weeks.append(date_tuple[1])
print(weeks)

#服务器列表，其中的元素为某规格服务器每天使用的情况
flavor_list=[]
for i in range(1,len(data_list[0])):
    flavor_list.append([data[i] for data in data_list])


#绘图
line_chart=pygal.Line(x_label_rotation=40,show_minor_x_labels=True)
line_chart.title='各规格服务器日使用情况(用例train_dataset)'
line_chart.x_labels=dates
N=7
line_chart._x_labels_major=dates[::N]
for i in range(1,len(data_list[0])):
    # print(i)
    line_name='flavor'+str(i)
    line_chart.add(line_name,flavor_list[i-1])
line_chart.render_to_file('flavor_used_day_month1~5.avg')