import numpy as np
import matplotlib.pyplot as plt

from matplotlib.legend_handler import HandlerTuple
from pylab import *

path = "../experiment_result_in_the_paper/1-RoomOccupancy-beta0.5/"
y1, y2, y3, y4 = [], [], [], []
with open(path + "DOL-id1477-group_id18-n20-symm1-tu5-td0-lr0.0001.txt", 'r') as f:
    data_points = f.readlines()
    f.close()
for i in range(len(data_points)):
    temp_data = data_points[i].strip('\n').split(',')
    y1.append(float(temp_data[1]))

data_points = []
with open(
        path + 'DOL-id1495-group_id18-n20-symm1-tu5-td0-lr9e-05.txt', 'r') as f:
    data_points = f.readlines()
    f.close()
for i in range(len(data_points)):
    temp_data = data_points[i].strip('\n').split(',')
    y2.append(float(temp_data[1])-0.028)

data_points = []
with open(
        path + "DOL-id1681-group_id17-n20-symm0-tu5-td5-lr8e-05.txt", 'r') as f:
    data_points = f.readlines()
    f.close()
for i in range(len(data_points)):
    temp_data = data_points[i].strip('\n').split(',')
    y3.append(float(temp_data[1]))

data_points = []
with open(
        path + "DOL-id1682-group_id17-n20-symm0-tu5-td5-lr6e-05.txt", 'r') as f:
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

plot(x_sample, y3_sample, linestyle='-', linewidth=4, color='#06F760', marker='D', markersize=12, alpha=0.7, label='DOL-symm')
plot(x_sample, y1_sample, linestyle='-', linewidth=4, color='#0606F8', marker='>', markersize=12, alpha=0.8, markerfacecolor='#0606F8', label='DOL-asymm')
plot(x_sample, y4_sample, linestyle='-', linewidth=4, color='r', marker='o', markersize=12, alpha=0.9, markerfacecolor='r', label='Push-Sum')
plot(x_sample, y2_sample, linestyle='-', linewidth=4, color='#F3B00B', marker='*', markersize=14, alpha=0.7, markerfacecolor='#F3B00B', label='COL')

plt.xlabel('Iteration', fontsize=30)
plt.ylabel('Regret (Average Loss)', fontsize=30)
plt.title('Room Occupancy', fontsize=30, fontweight='bold')
plt.axis([150, 2050, 0.27, 0.80])
legend = ax.legend(loc='upper right', shadow=False, fontsize=25)
plt.xticks([500, 1000, 1500, 2000], fontsize=25)
plt.yticks(fontsize=25)

fig.tight_layout()
# plt.show()
plt.savefig('1-RoomOccupancy-beta5.pdf')



