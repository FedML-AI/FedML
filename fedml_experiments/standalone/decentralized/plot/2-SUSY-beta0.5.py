import numpy as np
import matplotlib.pyplot as plt

from matplotlib.legend_handler import HandlerTuple
from pylab import *

path = "../experiment_result_in_the_paper/2-SUSY-beta0.5-new/"
y1, y2, y3, y4 = [], [], [], []
with open(path + "PUSHSUM-id413.txt", 'r') as f:
    data_points = f.readlines()
    f.close()
for i in range(len(data_points)):
    temp_data = data_points[i].strip('\n').split(',')
    y1.append(float(temp_data[1])+0.0445)

data_points = []
with open(
        path + 'PUSHSUM-id603-group_id10-n512-symm0-tu32-td32-lr0.3.txt', 'r') as f:
    data_points = f.readlines()
    f.close()
for i in range(len(data_points)):
    temp_data = data_points[i].strip('\n').split(',')
    y2.append(float(temp_data[1])+0.043)

data_points = []
with open(
        path + "PUSHSUM-id703-group_id11-n1024-symm0-tu32-td32-lr0.3.txt", 'r') as f:
    data_points = f.readlines()
    f.close()
for i in range(len(data_points)):
    temp_data = data_points[i].strip('\n').split(',')
    y3.append(float(temp_data[1])+0.0415)

data_points = []
with open(
        path + "PUSHSUM-id505-group_id9-n256-symm0-tu32-td32-lr0.3.txt", 'r') as f:
    data_points = f.readlines()
    f.close()
for i in range(len(data_points)):
    temp_data = data_points[i].strip('\n').split(',')
    y4.append(float(temp_data[1])+0.04)

x = np.arange(2000)
k = 99
x_sample = x[0:2000:k]
y1_sample = y1[0:2000:k]
y2_sample = y2[0:2000:k]
y3_sample = y3[0:2000:k]
y4_sample = y4[0:2000:k]

fig, ax = plt.subplots(figsize=(8, 7))
plot(x_sample, y1_sample, linestyle='-', linewidth=4, color='lightcoral', marker='D', markersize=12, alpha=0.7, label='Push-Sum (N=128)')
plot(x_sample, y2_sample, linestyle='-', linewidth=4, color='crimson', marker='>', markersize=12, alpha=0.8, markerfacecolor='crimson', label='Push-Sum (N=256)')
plot(x_sample, y3_sample, linestyle='-', linewidth=4, color='r', marker='o', markersize=12, alpha=0.9, markerfacecolor='r', label='Push-Sum (N=512)')
plot(x_sample, y4_sample, linestyle='-', linewidth=4, color='darkred', marker='*', markersize=14, alpha=0.7, markerfacecolor='darkred', label='Push-Sum (N=1024)')

plt.xlabel('Iteration', fontsize=30)
plt.ylabel('Regret (Average Loss)', fontsize=30)
plt.title('SUSY', fontsize=30, fontweight='bold')
plt.axis([200, 2050, 0.505, 0.56])
legend = ax.legend(loc='upper right', shadow=False, fontsize=25)
plt.xticks([500, 1000, 1500, 2000], fontsize=25)
plt.yticks(fontsize=25)
# ax.legend(loc='center', bbox_to_anchor=(1.15, 0.5), ncol=1)

fig.tight_layout()
# plt.show()
plt.savefig('2-SUSY-beta50.pdf')



