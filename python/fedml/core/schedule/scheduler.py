import numpy as np


class scheduler:
    def __init__(self, workloads, constraints, memory,
                uniform_client=True, uniform_gpu=False):
        self.workloads = workloads
        self.x = np.sort(workloads)[::-1]
        self.x_sorted_index = np.argsort(workloads)[::-1]
        self.y = constraints
        self.m = memory
        self.len_x = len(workloads)
        self.len_y = len(constraints)
        self.uniform_client = uniform_client 
        self.uniform_gpu = uniform_gpu


    def assign_a_workload_serial(self, x_maps, cost_maps):
        # Find the case with the minimum cost.
        costs = []
        for i in range(len(cost_maps)):
            costs.append(max(cost_maps[i]))
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
        for i in range(self.len_y):
            new_maps.append(np.copy(x_map))
            new_maps[i][target_index] = i
            new_costs.append(np.copy(cost_map))
            new_costs[i][i] += self.y[i] * self.x[target_index]

        # Insert all the new maps.
        for i in range(self.len_y):
            # Check if this case violates the memory constraints.
            max_cost = max(new_costs[i])
            resource_index = np.argmax(new_costs[i])
            if max_cost <= self.m[resource_index]:
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
        if mode == 1:
            resource_maps = []
            resource_maps.append(np.zeros((self.len_y)))
            x_maps, cost_maps, resource_maps = self.assign_a_workload(
                x_maps, cost_maps, resource_maps
            )
        else:
            x_maps, cost_maps = self.assign_a_workload_serial(x_maps, cost_maps)

        costs = []
        for i in range(len(cost_maps)):
            costs.append(max(cost_maps[i]))
        target_index = np.argmin(costs)

        schedules = []
        for i in range(self.len_y):
            my_jobs = []
            for j in range(self.len_x):
                if x_maps[target_index][j] == i:
                    my_jobs.append(self.x_sorted_index[j])
            schedules.append(my_jobs)
        print(
            "The optimal maximum cost: %d, assignment: %s\n"
            % (costs[target_index], str(x_maps[target_index]))
        )

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
        return output_schedules


if __name__ == "__main__":
    mode = 1
    workloads = np.array([1, 2, 3, 5, 7, 14])
    constraints = np.array([1, 5])
    memory = np.array([15, 100])
    my_scheduler = scheduler(workloads, constraints, memory)
    schedules = my_scheduler.DP_schedule(mode)
    for i in range(len(schedules)):
        print("Resource %2d: %s\n" % (i, str(schedules[i])))
