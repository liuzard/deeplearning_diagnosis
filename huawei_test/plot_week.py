import pygal
from datetime import datetime
from huawei_test.file_reader_2 import file_trans
from itertools import groupby
import xlwt


test_file_name="C:\\Users\liuzard\PycharmProjects\deep_learning\huawei_test\练习数据\data_2015_1-5.txt"
#读取文件数据，并将数据存储到列表中
data_list=file_trans(test_file_name)

#获取数据列表中的日期
dates=[date[0] for date in data_list]

#获取日期所在周数的列表
weeks=[]
for date in dates:
    dl=datetime.strptime(date,'%Y-%m-%d')
    date_tuple=dl.isocalendar()
    week='第'+str(date_tuple[1])+'周'
    weeks.append(week)
print(len(weeks))

#服务器列表，其中的元素为某规格服务器每天使用的情况
flavor_list=[]
for i in range(1,len(data_list[0])):
    flavor_list.append([data[i] for data in data_list])


file = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = file.add_sheet('trian_data')
for i in range(len(flavor_list)):
    xy_map=[]
    for x,y in groupby(zip(weeks,flavor_list[i]),key=lambda _:_[0]):
        y_list=[v for _,v in y]
        xy_map.append([x,sum(y_list)/len(y_list)])
    print(xy_map)
    x_unique,y_mean=zip(*xy_map)
    print(y_mean)
    # 新建一个excel文件

    # 新建一个sheet
    for j in range(len(y_mean)):
        sheet.write(j, i, y_mean[j])
file.save('train_data_week.xls')








def draw_line_forweek(x_data,y_data,y_legend,line_chart):
    xy_map=[]
    for x,y in groupby(zip(x_data,y_data),key=lambda _:_[0]):
        y_list=[v for _,v in y]
        xy_map.append([x,sum(y_list)/len(y_list)])
    print(xy_map)
    x_unique,y_mean=[*zip(*xy_map)]
    # line_chart=pygal.Line()
    # line_chart.title=title
    line_chart.x_labels=x_unique
    line_chart.add(y_legend,y_mean)
    # line_chart.render_to_file(title+'.svg')
    return line_chart



#绘图
title="各规格服务器日均使用情况"
line_chart=pygal.Line()
line_chart=pygal.Line(x_label_rotation=40,show_minor_x_labels=True)
for i in range(1,len(data_list[0])):
    x_inputs=weeks
    y_inputs=flavor_list[i-1]
    y_legend='flavor'+str(i)
    line_chart=draw_line_forweek(x_inputs,y_inputs,y_legend,line_chart)
line_chart.title=title
line_chart.render_to_file('flavor_used_week_month1-5.avg')
