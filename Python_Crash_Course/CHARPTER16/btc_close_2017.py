from __future__ import (absolute_import,division,print_function,unicode_literals)
from urllib.request import urlopen
import json
import requests
import pygal
import math
from itertools import groupby

json_url='http://raw.githubusercontent.com/muxuezi/btc/master/btc_close_2017.json'
'''
读json数据 
#利用urlopen 读取json 数据
response=urlopen(json_url)
req1=response.read()
with open('btc_close_2017_urllib.json','wb') as f:
    f.write(req1)
file_urllib=json.loads(req1.decode('utf-8'))
print(file_urllib)

#利用第三方模块request读取json数据
req2=requests.get(json_url)
with open('btc_close_2017_request.json','w') as f:
    f.write(req2.text)
file_request=req2.json()
print(file_request)
print(file_urllib==file_request)
'''

filename='btc_close_2017_urllib.json'
with open(filename) as f:
    btc_data=json.load(f)


dates,months,weeks,weekdays,close=[],[],[],[],[]
for btc_dict in btc_data:
    dates.append(btc_dict['date'])
    months.append(btc_dict['month'])
    weeks.append(btc_dict['week'])
    weekdays.append(btc_dict['weekday'])
    close.append(int(float(btc_dict['close'])))
print(dates)

def draw_line(x_data,y_data,title,y_legend):
    xy_map=[]
    for x,y in groupby(sorted(zip(x_data,y_data)),key=lambda _:_[0]):
        y_list=[v for _,v in y]
        xy_map.append([x,sum(y_list)/len(y_list)])

    x_unique,y_mean=[*zip(*xy_map)]
    print(x_unique)
    line_chart=pygal.Line()
    line_chart.title=title
    line_chart.x_labels=x_unique
    line_chart.add(y_legend,y_mean)
    line_chart.render_to_file(title+'.svg')
    return line_chart

def draw_line_forweek(x_data,y_data,title,y_legend):
    xy_map=[]
    for x,y in groupby(zip(x_data,y_data),key=lambda _:_[0]):
        y_list=[v for _,v in y]
        xy_map.append([x,sum(y_list)/len(y_list)])

    x_unique,y_mean=[*zip(*xy_map)]
    print(x_unique)
    line_chart=pygal.Line()
    line_chart.title=title
    line_chart.x_labels=x_unique
    line_chart.add(y_legend,y_mean)
    line_chart.render_to_file(title+'.svg')
    return line_chart


idx_month=dates.index('2017-12-01')
line_chart_month=draw_line(months[:idx_month],close[:idx_month],title='收盘价单月日均值',y_legend='月日均值')

idx_week=dates.index('2017-12-11')
line_chart_week=draw_line_forweek(weeks[1:idx_week],close[1:idx_week],title="收盘价每周日均值",y_legend='每周日均值')

idx_in_week=dates.index('2017-12-11')
wd=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
weekdays_int=[wd.index(w) for w in weekdays[1:idx_in_week]]
line_chart_weekday=draw_line(weekdays_int,close[1:idx_in_week],title='收盘价星期均值',y_legend='星期均值')
line_chart_weekday.x_labels=['周一','周二','周三','周四','周五','周六','周日']
line_chart_weekday.render_to_file('收盘价星期均值.svg')


print(dates.index('2017-12-01'))
line_chart=pygal.Line(x_label_rotation=20,show_minor_x_labels=False)
line_chart.title='收盘价（￥）'
line_chart.x_labels=dates
N=20#x轴坐标每隔20天显示一次
line_chart.x_labels_major=dates[::N]
close_log=[math.log10(x) for x in close]
line_chart.add('log收盘价',close_log)
line_chart.render_to_file('收盘价对数变换折线图（￥）.svg')