import numpy as np


def linear_fit(x, y):
    z1 = np.polyfit(x, y, 1)
    p1 = np.poly1d(z1)
    print(p1)
    yvals = p1(x)
    # fit_error = np.abs(yvals - y)
    fit_dif = np.abs(yvals - y)
    fit_error = np.mean(fit_dif / y)
    # fit_error = np.linalg.norm(yvals - y)
    return z1, p1, yvals, fit_error


def t_sample_fit(
    num_workers, num_clients, runtime_history, train_data_local_num_dict, uniform_client=False, uniform_gpu=False
):
    """
        runtime_history: {
            0: {0: [], 1: [], 2: []...},
            1: {0: [], 1: [], 2: []...},
        }  
    """
    fit_params = {}
    fit_funcs = {}
    fit_errors = {}

    runtime_to_fit = {}
    data_local_num_dict = {}
    if uniform_client and uniform_gpu:
        runtime_to_fit[0] = {}
        runtime_to_fit[0][0] = []
        data_local_num_dict[0] = {}
        data_local_num_dict[0][0] = []
        for worker_id in range(num_workers):
            for client_id in range(num_clients):
                runtime_info = runtime_history[worker_id][client_id]
                if isinstance(runtime_info, list):
                    runtime_to_fit[0][0] += runtime_info
                    data_local_num_dict[0][0] += [train_data_local_num_dict[client_id]] * len(runtime_info)
                elif runtime_info is None:
                    pass
                elif runtime_info > 0:
                    runtime_to_fit[0][0].append(runtime_info)
                    data_local_num_dict[0][0] += [train_data_local_num_dict[client_id]]

    elif not uniform_client and uniform_gpu:
        runtime_to_fit[0] = {}
        data_local_num_dict[0] = {}
        for worker_id in range(num_workers):
            for client_id in range(num_clients):
                if client_id not in runtime_to_fit[0]:
                    runtime_to_fit[0][client_id] = []
                    data_local_num_dict[0][client_id] = []
                runtime_info = runtime_history[worker_id][client_id]
                if isinstance(runtime_info, list):
                    runtime_to_fit[0][client_id] += runtime_info
                    data_local_num_dict[0][client_id] += [train_data_local_num_dict[client_id]] * len(runtime_info)
                elif runtime_info is None:
                    pass
                elif runtime_info > 0:
                    runtime_to_fit[0][client_id].append(runtime_info)
                    data_local_num_dict[0][client_id] += [train_data_local_num_dict[client_id]]

    elif uniform_client and not uniform_gpu:
        for worker_id in range(num_workers):
            runtime_to_fit[worker_id] = {}
            runtime_to_fit[worker_id][0] = []
            data_local_num_dict[worker_id] = {}
            data_local_num_dict[worker_id][0] = []
            for client_id in range(num_clients):
                runtime_info = runtime_history[worker_id][client_id]
                if isinstance(runtime_info, list):
                    runtime_to_fit[worker_id][0] += runtime_info
                    data_local_num_dict[worker_id][0] += [train_data_local_num_dict[client_id]] * len(runtime_info)
                elif runtime_info is None:
                    pass
                elif runtime_info > 0:
                    runtime_to_fit[worker_id][0].append(runtime_info)
                    data_local_num_dict[worker_id][0] += [train_data_local_num_dict[client_id]]
    else:
        for worker_id in range(num_workers):
            runtime_to_fit[worker_id] = {}
            data_local_num_dict[worker_id] = {}
            for client_id in range(num_clients):
                if client_id not in runtime_to_fit[worker_id]:
                    runtime_to_fit[worker_id][client_id] = []
                    data_local_num_dict[worker_id][client_id] = []
                runtime_info = runtime_history[worker_id][client_id]
                if isinstance(runtime_info, list):
                    runtime_to_fit[worker_id][client_id] += runtime_info
                    data_local_num_dict[worker_id][client_id] += [train_data_local_num_dict[client_id]] * len(
                        runtime_info
                    )
                elif runtime_info is None:
                    pass
                elif runtime_info > 0:
                    runtime_to_fit[worker_id][client_id].append(runtime_info)
                    data_local_num_dict[worker_id][client_id] += [train_data_local_num_dict[client_id]]

    # logging.info(f"runtime_to_fit: {runtime_to_fit}")
    # logging.info(f"data_local_num_dict: {data_local_num_dict}")
    for worker_id, runtime_on_clients in runtime_to_fit.items():
        fit_params[worker_id] = {}
        fit_funcs[worker_id] = {}
        fit_errors[worker_id] = {}
        for client_id, runtimes in runtime_on_clients.items():
            x = data_local_num_dict[worker_id][client_id]
            z1, p1, yvals, fit_error = linear_fit(x=data_local_num_dict[worker_id][client_id], y=runtimes)
            fit_params[worker_id][client_id] = z1
            fit_funcs[worker_id][client_id] = p1
            fit_errors[worker_id][client_id] = fit_error
    return fit_params, fit_funcs, fit_errors


if __name__ == "__main__":
    num_workers = 4
    num_clients = 5
    train_data_local_num_dict = {0: 100, 1: 200, 2: 300, 3: 400, 4: 250}
    gpu_power = {0: 1, 1: 1, 2: 1, 3: 1}

    runtime_history = {}
    for i in range(num_workers):
        runtime_history[i] = {}
        for j in range(num_clients):
            runtime_history[i][j] = (train_data_local_num_dict[j] + 10 * np.random.rand(3)).tolist()

    fit_params, fit_funcs, fit_errors = t_sample_fit(
        4, 5, runtime_history, train_data_local_num_dict, uniform_client=True, uniform_gpu=False
    )

    print(fit_params)
    print(fit_funcs)
    print(fit_errors)
