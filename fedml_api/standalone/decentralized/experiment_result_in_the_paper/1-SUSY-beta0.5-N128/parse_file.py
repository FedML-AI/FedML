#path = "../experiment_result_in_the_paper/2-SUSY-beta0/"
iteration_list = []
loss_list = []

f_read = open("471", 'r')
f_write = open("PUSHSUM-id471-group_id8-n128-symm0-tu16-td16-lr0.4.txt", 'w+')
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
