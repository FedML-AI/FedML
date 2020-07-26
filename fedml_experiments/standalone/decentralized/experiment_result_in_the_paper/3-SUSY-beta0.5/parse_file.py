iteration_list = []
loss_list = []

f_read = open("1101", 'r')
f_write = open("PUSHSUM-id1101-group_id15-n512-symm0-tu200-td200-lr0.1.txt", 'w+')
iteration_index = 0
data_points = f_read.readlines()
for i in range(len(data_points)):
    if data_points[i][0:6] == 'regret':

        temp_data = data_points[i].strip('\n').split(',')
        loss = float(temp_data[0][-6:])
        f_write.write(str(iteration_index) + "," + str(loss)+"\n")
        iteration_index+=1
f_read.close()
f_write.close()
