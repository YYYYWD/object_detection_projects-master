import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

plt.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块


d = pd.DataFrame.from_csv('report.csv', index_col=None)
#d['5 seconds'] = (d['time']/int(5)*100).astype(int)
#d = d.groupby('5 seconds').agg({'vehicles':np.sum,"time":lambda x: x.iloc[0]})
#d['time'] = (d['time']/100).astype(int)
#d['time'] = pd.to_datetime(d['time'],unit='s')
#d = d.set_index(['time'], drop=True)
#d.plot()
#plt.show()
print(d)
count=0
all_number=0
countlist=[]
allcount=[]
t=0
for i in d['vehicles']:
    count =int(i)+count
    all_number=int(i)+all_number
    if(t%15==0):
        countlist.append(count)
        allcount.append(all_number)
        count=0

    t=t+1

time=range(0,len(countlist))
print(list(time))
print(allcount)

plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.xlim((0, len(time)))

plt.plot(time,allcount,color='red',linewidth=1.0)
plt.xlabel('time /s')
plt.ylabel('all_count')
plt.grid(True)
plt.legend('车')




plt.subplot(122)
#plt.plot(time,countlist)
plt.bar(time,countlist)
plt.xlim((0, len(time)))
plt.xlabel('time /s')
plt.ylabel('count')
plt.grid(True)
plt.legend('车辆')
plt.savefig("result.jpg")  #保存图象
plt.show()

# else:
#     print ("Usage: python plot.py [path to the csv report] [number of seconds to group by]")
