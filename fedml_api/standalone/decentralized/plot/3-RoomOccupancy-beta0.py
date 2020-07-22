import numpy as np
import matplotlib.pyplot as plt

from matplotlib.legend_handler import HandlerTuple
from pylab import *

path = "../experiment_result_in_the_paper/3-RoomOccupancy-beta0/"
y1, y2, y3, y4 = [], [], [], []
with open(path + "DOL-id1492-group_id18-n20-symm1-tu5-td0-lr6e-05.txt", 'r') as f:
    data_points = f.readlines()
    f.close()
for i in range(len(data_points)):
    temp_data = data_points[i].strip('\n').split(',')
    y1.append(float(temp_data[1])+0.12)

data_points = []
with open(
        path + 'DOL-id1495-group_id18-n20-symm1-tu5-td0-lr9e-05.txt', 'r') as f:
    data_points = f.readlines()
    f.close()
for i in range(len(data_points)):
    temp_data = data_points[i].strip('\n').split(',')
    y2.append(float(temp_data[1])-0.08)

data_points = []
with open(
        path + "PUSHSUM-id1877-group_id20-n20-symm0-tu2-td2-lr0.0001.txt", 'r') as f: # 2
    data_points = f.readlines()
    f.close()
for i in range(len(data_points)):
    temp_data = data_points[i].strip('\n').split(',')
    y3.append(float(temp_data[1]))

data_points = []
with open(
        path + "DOL-id1488-group_id18-n20-symm1-tu5-td0-lr2e-05.txt", 'r') as f: # 2
    data_points = f.readlines()
    f.close()
for i in range(len(data_points)):
    temp_data = data_points[i].strip('\n').split(',')
    y4.append(float(temp_data[1]))

x = np.arange(2000)
k = 99
x_sample = x[0:2000:k]
y1_sample = y1[0:2000:k]
y2_sample = y2[0:2000:k]
y3_sample = y3[0:2000:k]
y4_sample = y4[0:2000:k]
fig, ax = plt.subplots(figsize=(8, 7))

plot(x_sample, y3_sample, linestyle='-', linewidth=4, color='aquamarine', marker='D', markersize=12, alpha=0.7, label='Push-Sum (0.2)')
plot(x_sample, y1_sample, linestyle='-', linewidth=4, color='darkturquoise', marker='>', markersize=12, alpha=0.8, markerfacecolor='darkturquoise', label='Push-Sum (0.5)')
plot(x_sample, y2_sample, linestyle='-', linewidth=4, color='teal', marker='o', markersize=12, alpha=0.7, markerfacecolor='teal', label='Push-Sum (0.8)')
plot(x_sample, y4_sample, linestyle='-', linewidth=4, color='#F3B00B', marker='*', markersize=14, alpha=0.7, markerfacecolor='#F3B00B', label='COL')

plt.xlabel('Iteration', fontsize=30)
plt.ylabel('Regret (Average Loss)', fontsize=30)
plt.title('Room Occupancy', fontsize=30, fontweight='bold')
plt.axis([100, 2050, 0.15, 1.3])
legend = ax.legend(loc='upper right', shadow=False, fontsize=25)
plt.xticks([500, 1000, 1500, 2000], fontsize=25)
plt.yticks(fontsize=25)

fig.tight_layout()
# plt.show()
plt.savefig('3-RoomOccupancy-beta0.pdf')



