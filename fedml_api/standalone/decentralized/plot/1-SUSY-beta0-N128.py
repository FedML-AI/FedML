import numpy as np
import matplotlib.pyplot as plt

from matplotlib.legend_handler import HandlerTuple
from pylab import *

path = "../experiment_result_in_the_paper/1-SUSY-beta0-N128/"
y1, y2, y3, y4 = [], [], [], []
with open(path + "DOL-id115-group_id1-n128-symm1-tu128-td0-lr0.5.txt", 'r') as f:
    data_points = f.readlines()
    f.close()
for i in range(len(data_points)):
    temp_data = data_points[i].strip('\n').split(',')
    y1.append(float(temp_data[1]))

data_points = []
with open(
        path + 'DOL-id217-group_id2-n128-symm1-tu16-td0-lr0.25.txt', 'r') as f:
    data_points = f.readlines()
    f.close()
for i in range(len(data_points)):
    temp_data = data_points[i].strip('\n').split(',')
    y2.append(float(temp_data[1]))

data_points = []
with open(
        path + "DOL-id314-group_id3-n128-symm0-tu16-td16-lr0.5.txt", 'r') as f:
    data_points = f.readlines()
    f.close()
for i in range(len(data_points)):
    temp_data = data_points[i].strip('\n').split(',')
    y3.append(float(temp_data[1]))

data_points = []
with open(
        path + "PUSHSUM-id413-group_id4-n128-symm0-tu16-td16-lr0.4.txt", 'r') as f:
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
plot(x_sample, y2_sample, linestyle='-', linewidth=4, color='#06F760', marker='D', markersize=12, alpha=0.7, label='DOL-symm')
plot(x_sample, y3_sample, linestyle='-', linewidth=4, color='#0606F8', marker='>', markersize=12, alpha=0.8, markerfacecolor='#0606F8', label='DOL-asymm')
plot(x_sample, y4_sample, linestyle='-', linewidth=4, color='r', marker='o', markersize=12, alpha=0.9, markerfacecolor='r', label='Push-Sum')
plot(x_sample, y1_sample, linestyle='-', linewidth=4, color='#F3B00B', marker='*', markersize=14, alpha=0.7, markerfacecolor='#F3B00B', label='COL')

plt.xlabel('Iteration', fontsize=30)
plt.ylabel('Regret (Average Loss)', fontsize=30)
plt.title('SUSY', fontsize=30, fontweight='bold')
plt.axis([500, 2050, 0.468, 0.52])
legend = ax.legend(loc='upper right', shadow=False, fontsize=25)
plt.xticks([500, 1000, 1500, 2000], fontsize=25)
plt.yticks(fontsize=25)
# ax.legend(loc='center', bbox_to_anchor=(1.15, 0.5), ncol=1)

fig.tight_layout()
# plt.show()
plt.savefig('1-SUSY-beta100-N128.pdf')



