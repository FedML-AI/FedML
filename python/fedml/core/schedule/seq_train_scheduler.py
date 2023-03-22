import logging
import sys

import numpy as np

sys.setrecursionlimit(10000)


class SeqTrainScheduler:
    def __init__(
        self,
        workloads,
        constraints,
        memory,
        cost_funcs,
        uniform_client=True,
        uniform_gpu=False,
        prune_equal_sub_solution=True,
    ):
        self.workloads = workloads
        self.x = np.sort(workloads)[::-1]
        self.x_sorted_index = np.argsort(workloads)[::-1]
        # print(f"self.x_sorted_index: {self.x_sorted_index} len(self.x_sorted_index): {len(self.x_sorted_index)}")
        self.y = constraints
        # logging.info(f"self.workloads: {self.workloads}, self.y: {self.y}")
        self.m = memory
        self.cost_funcs = cost_funcs
        self.uniform_client = uniform_client
        self.uniform_gpu = uniform_gpu
        self.len_x = len(workloads)
        self.len_y = len(constraints)
        self.prune_equal_sub_solution = prune_equal_sub_solution
        self.iter_times = 0

    def obtain_client_cost(self, resource_id, client_id):
        if self.uniform_client and self.uniform_gpu:
            # cost = self.cost_funcs[0][0](self.client_data_nums[client_id])
            cost = self.cost_funcs[0][0](self.workloads[client_id])
        elif not self.uniform_client and self.uniform_gpu:
            # cost = self.cost_funcs[0][client_id](self.client_data_nums[client_id])
            cost = self.cost_funcs[0][client_id](self.workloads[client_id])
        elif self.uniform_client and not self.uniform_gpu:
            # cost = self.cost_funcs[resource_id][0](self.client_data_nums[client_id])
            cost = self.cost_funcs[resource_id][0](self.workloads[client_id])
        else:
            # cost = self.cost_funcs[resource_id][client_id](self.client_data_nums[client_id])
            cost = self.cost_funcs[resource_id][client_id](self.workloads[client_id])
        if cost < 0.0:
            cost = 0.0
        return cost

    def assign_a_workload_serial(self, x_maps, cost_maps):
        # Find the case with the minimum cost.
        self.iter_times += 1
        costs = []
        for i in range(len(cost_maps)):
            costs.append(max(cost_maps[i]))
        costs = np.array(costs)
        # logging.info(f"self.iter_times: {self.iter_times}, cost_maps:{cost_maps} ")
        # delete other items that are not optimimal sub combinations.
        if self.prune_equal_sub_solution:
            target_case_index = np.argmin(costs)
            costs = [costs[target_case_index]]
            cost_maps = [cost_maps[target_case_index]]
            x_maps = [x_maps[target_case_index]]
        else:
            min_indexes = np.argwhere(costs == np.amin(costs))
            min_indexes = [i[0] for i in min_indexes]
            costs = costs[min_indexes]
            cost_maps = [cost_maps[i] for i in min_indexes]
            x_maps = [x_maps[i] for i in min_indexes]
        target_case_index = np.argmin(costs)

        # Check if the minimum cost has the full map.
        x_map = x_maps[target_case_index]
        if x_map[-1] >= 0:
            return x_maps, cost_maps

        # Find the workload index we will work on now.
        x_map = x_maps.pop(target_case_index)
        cost_map = cost_maps.pop(target_case_index)

        target_index = 0
        for i in range(len(x_map)):
            if x_map[i] == -1:
                target_index = i
                break

        # Create len_y maps.
        new_maps = []
        new_costs = []
        client_id = self.x_sorted_index[target_index]
        for i in range(self.len_y):
            new_maps.append(np.copy(x_map))
            new_maps[i][target_index] = i
            new_costs.append(np.copy(cost_map))
            #             new_costs[i][i] += self.y[i] * self.x[target_index]
            new_costs[i][i] += self.obtain_client_cost(i, client_id)

        # Insert all the new maps.
        for i in range(self.len_y):
            # Check if this case violates the memory constraints.
            max_cost = max(new_costs[i])
            resource_index = np.argmax(new_costs[i])
            #             if max_cost <= self.m[resource_index]:
            x_maps.append(new_maps[i])
            cost_maps.append(new_costs[i])
        return self.assign_a_workload_serial(x_maps, cost_maps)

    def assign_a_workload(self, x_maps, cost_maps, resource_maps):
        # Find the case with the minimum cost.
        costs = []
        for i in range(len(cost_maps)):
            costs.append(max(cost_maps[i]))
        target_case_index = np.argmin(costs)

        # Check if the minimum cost has the full map.
        x_map = x_maps[target_case_index]
        if x_map[-1] >= 0:
            return x_maps, cost_maps, resource_maps

        # Find the workload index we will work on now.
        x_map = x_maps.pop(target_case_index)
        cost_map = cost_maps.pop(target_case_index)
        resource_map = resource_maps.pop(target_case_index)

        target_index = 0
        for i in range(len(x_map)):
            if x_map[i] == -1:
                target_index = i
                break

        # Create len_y maps.
        new_maps = []
        new_costs = []
        new_resources = []
        for i in range(self.len_y):
            # Parallel run.
            new_maps.append(np.copy(x_map))
            new_maps[-1][target_index] = i
            new_costs.append(np.copy(cost_map))
            new_costs[-1][i] = max((self.y[i] * self.x[target_index]), new_costs[-1][i])
            new_resources.append(np.copy(resource_map))
            new_resources[-1][i] += self.x[target_index]

            # Serial run.
            new_maps.append(np.copy(x_map))
            new_maps[-1][target_index] = i
            new_costs.append(np.copy(cost_map))
            new_costs[-1][i] += self.y[i] * self.x[target_index]
            new_resources.append(np.copy(resource_map))
            new_resources[-1][i] = self.x[target_index]

        # Insert all the new maps.
        for i in range(len(new_resources)):
            max_mem = max(new_resources[i])
            resource_index = np.argmax(new_resources[i])
            if max_mem <= self.m[resource_index]:
                x_maps.append(new_maps[i])
                # print ("max_mem of resource %d: %d cost: %d %s\n" %(resource_index, max_mem, max(new_costs[i]), str(new_maps[i])))
                cost_maps.append(new_costs[i])
                resource_maps.append(new_resources[i])
        return self.assign_a_workload(x_maps, cost_maps, resource_maps)

    def DP_schedule(self, mode):
        x_maps = []
        x_maps.append(np.negative(np.ones((self.len_x))))
        cost_maps = []
        cost_maps.append(np.zeros((self.len_y)))
        # print(f"Initial x_maps: {x_maps} len(x_maps): {len(x_maps)}")
        # print(f"Initial cost_maps: {cost_maps}  len(cost_maps): {len(cost_maps)}")
        if mode == 1:
            resource_maps = []
            resource_maps.append(np.zeros((self.len_y)))
            x_maps, cost_maps, resource_maps = self.assign_a_workload(x_maps, cost_maps, resource_maps)
        else:
            x_maps, cost_maps = self.assign_a_workload_serial(x_maps, cost_maps)

        # print(f"x_maps: {x_maps} len(x_maps): {len(x_maps)}")
        # print(f"cost_maps: {cost_maps}  len(cost_maps): {len(cost_maps)}")
        costs = []
        for i in range(len(cost_maps)):
            costs.append(max(cost_maps[i]))
        target_index = np.argmin(costs)
        # print(f"target_index: {target_index} ")

        schedules = []
        for i in range(self.len_y):
            my_jobs = []
            for j in range(self.len_x):
                if x_maps[target_index][j] == i:
                    my_jobs.append(self.x_sorted_index[j])
            schedules.append(my_jobs)

        # logging.info(f"schedules: {schedules}  len(schedules): {len(schedules)}")
        logging.info(f"self.iter_times: {self.iter_times}")
        logging.info(
            "The optimal maximum cost: %f, assignment: %s\n" % (costs[target_index], str(x_maps[target_index]))
        )
        logging.info(f"target_index: {target_index} cost_map: {cost_maps[target_index]}")

        # print(f"schedules: {schedules}  len(schedules): {len(schedules)}")
        # print(f"self.iter_times: {self.iter_times}")
        # print(
        #     "The optimal maximum cost: %f, assignment: %s\n"
        #     % (costs[target_index], str(x_maps[target_index]))
        # )
        # print(f"target_index: {target_index} cost_map: {cost_maps[target_index]}")

        if mode == 1:
            output_schedules = []
            for i in range(len(schedules)):
                schedule = {}
                jobs = []
                sequence = schedules[i]
                footprint = 0
                for j in range(len(sequence)):
                    if footprint + self.x[j] <= self.m[i]:
                        jobs.append(sequence[j])
                        footprint += self.x[j]
                    else:
                        num_bunches = len(schedule)
                        schedule[num_bunches] = jobs
                        jobs = []
                        jobs.append(sequence[j])
                        footprint = self.x[j]
                if footprint > 0:
                    num_bunches = len(schedule)
                    schedule[num_bunches] = jobs
                output_schedules.append(schedule)
        else:
            output_schedules = []
            for i in range(len(schedules)):
                schedule = {}
                sequence = schedules[i]
                for j in range(len(sequence)):
                    jobs = [sequence[j]]
                    num_bunches = len(schedule)
                    schedule[num_bunches] = jobs
                output_schedules.append(schedule)
        return schedules, output_schedules

