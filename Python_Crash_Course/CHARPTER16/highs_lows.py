import csv
import matplotlib.pyplot as plt
from datetime import datetime

file_name='death_valley_2014.csv'
with open(file_name) as f:
    reader=csv.reader(f)
    header_row=next(reader)
    dates,highs,lows=[],[],[]
    for row in reader:
        try:
            date = datetime.strptime(row[0], '%Y-%m-%d')
            high = int(row[1])
            low = int(row[3])
        except ValueError:
            print(date,'missing data')
        else:
            highs.append(high)
            lows.append(low)
            dates.append(date)

    print(highs,dates)

fig=plt.figure(dpi=128,figsize=(10,6))
plt.plot(dates,highs,c='red',linewidth=1,alpha=0.5)
plt.plot(dates,lows,c='blue',linewidth=1,alpha=0.5)
plt.fill_between(dates,highs,lows,facecolor='blue',alpha=0.1)
plt.title("daily high temperatures,2014",fontsize=24)
plt.xlabel("",fontsize=1)
fig.autofmt_xdate()
plt.ylabel("Temperature(F)",fontsize=16)
plt.tick_params(axis='both',which='major',labelsize=20)

plt.show()