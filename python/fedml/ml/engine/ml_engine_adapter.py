import torch

from .torch_process_group_manager import TorchProcessGroupManager


def is_torch_device_available(args, device_type):
    if device_type == "gpu":
        if torch.cuda.is_available():
            return True
        return False
    elif device_type == "mps":
        # Macbook M1: https://pytorch.org/docs/master/notes/mps.html
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not " "built with MPS enabled.")
            else:
                print(
                    "MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine."
                )
            return False
        else:
            return True
    elif device_type == "cpu":
        return True

    return False


def is_device_available(args, device_type="gpu"):
    if hasattr(args, "ml_engine"):
        if args.ml_engine == "tf":
            import tensorflow as tf

            devices = tf.config.list_physical_devices(device_type.upper())
            if len(devices) > 0:
                return True
            return False
        elif args.ml_engine == "jax":
            try:
                import jax

                device_count = jax.device_count(device_type)
                if device_count > 0:
                    return True
            except Exception as ex:
                return False

            return False
        else:
            return is_torch_device_available(args, device_type)
    else:
        return is_torch_device_available(args, device_type)


def get_torch_device(args, using_gpu, device_id, device_type):
    if using_gpu:
        gpu_id = args.gpu_id
        if device_id is not None:
            gpu_id = device_id

        if torch.cuda.is_available() and device_type == "gpu":
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(gpu_id)
        elif device_type == "mps":
            # https://pytorch.org/docs/master/notes/mps.html
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        return device
    else:
        return torch.device("cpu")


def get_tf_device(args, using_gpu, device_id, device_type):
    import tensorflow as tf

    if using_gpu:
        return tf.device("/device:gpu:{}".format(device_id))
    else:
        return tf.device("/device:cpu:0")


def get_jax_device(args, using_gpu, device_id, device_type):
    import jax

    devices = jax.devices(None)
    if len(devices) > 0:
        for dev in devices:
            if dev.id == device_id:
                return dev
        return devices[0]
    else:
        return None


def get_device(args, using_gpu=False, device_id=None, device_type="cpu"):
    if hasattr(args, "using_gpu") and args.using_gpu:
        using_gpu = True

    if hasattr(args, "ml_engine"):
        if args.ml_engine == "tf":
            return get_tf_device(args, using_gpu, device_id, device_type)
        elif args.ml_engine == "jax":
            return get_jax_device(args, using_gpu, device_id, device_type)
        else:
            return get_torch_device(args, using_gpu, device_id, device_type)
    else:
        return get_torch_device(args, using_gpu, device_id, device_type)


def dict_to_device(args, dict_obj, device):
    if hasattr(args, "ml_engine"):
        if args.ml_engine == "tf":
            with device:
                dict_ret = dict_obj
                return dict_ret
        elif args.ml_engine == "jax":
            import jax

            return jax.device_put(dict_obj, device)
        else:
            return dict_obj.to(device)
    else:
        return dict_obj.to(device)


def model_params_to_device(args, params_obj, device):
    if hasattr(args, "ml_engine"):
        if args.ml_engine == "tf":
            with device:
                params_ret = params_obj
                return params_ret
        elif args.ml_engine == "jax":
            for key in params_obj.keys():
                params_obj[key] = dict_to_device(args, params_obj[key], device)
        else:
            for key in params_obj.keys():
                params_obj[key] = dict_to_device(args, params_obj[key], device)
    else:
        for key in params_obj.keys():
            params_obj[key] = dict_to_device(args, params_obj[key], device)

    return params_obj


def model_to_device(args, model_obj, device):
    if hasattr(args, "ml_engine"):
        if args.ml_engine == "tf":
            with device:
                model_ret = model_obj
                return model_ret
        elif args.ml_engine == "jax":
            return model_obj
        else:
            model_obj.to(device)
    else:
        model_obj.to(device)


def torch_model_ddp(args, model_obj, device):
    from torch.nn.parallel import DistributedDataParallel as DDP

    only_gpu = args.using_gpu
    process_group_manager = TorchProcessGroupManager(
        args.proc_rank_in_silo, args.n_proc_in_silo, args.pg_master_address, args.pg_master_port, only_gpu,
    )
    model = DDP(model_obj, device_ids=[device] if only_gpu else None)
    return process_group_manager, model


def tf_model_ddp(args, model_obj, device):
    process_group_manager, model = None, model_obj
    return process_group_manager, model


def jax_model_ddp(args, model_obj, device):
    process_group_manager, model = None, model_obj
    return process_group_manager, model


def model_ddp(args, model_obj, device):
    process_group_manager, model = None, model_obj
    if hasattr(args, "ml_engine"):
        if args.ml_engine == "tf":
            return tf_model_ddp(args, model_obj, device)
        elif args.ml_engine == "jax":
            return jax_model_ddp(args, model_obj, device)
        else:
            process_group_manager, model = torch_model_ddp(args, model_obj, device)
    else:
        process_group_manager, model = torch_model_ddp(args, model_obj, device)

    return process_group_manager, model


def torch_aggregator(args, raw_grad_list, training_num):
    (num0, avg_params) = raw_grad_list[0]

    if args.federated_optimizer == "FedAvg":
        for k in avg_params.keys():
            for i in range(0, len(raw_grad_list)):
                local_sample_number, local_model_params = raw_grad_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    avg_params[k] = local_model_params[k] * w
                else:
                    avg_params[k] += local_model_params[k] * w
    elif args.federated_optimizer == "FedAvg_seq":
        for k in avg_params.keys():
            for i in range(0, len(raw_grad_list)):
                local_sample_number, local_model_params = raw_grad_list[i]
                if i == 0:
                    avg_params[k] = local_model_params[k]
                else:
                    avg_params[k] += local_model_params[k]
    elif args.federated_optimizer == "FedOpt":
        pass

    return avg_params


def tf_aggregator(args, raw_grad_list, training_num):
    (num0, avg_params) = raw_grad_list[0]

    if args.federated_optimizer == "FedAvg":
        for k in range(0, len(avg_params)):
            for i in range(0, len(raw_grad_list)):
                local_sample_number, local_model_params = raw_grad_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    avg_params[k] = local_model_params[k] * w
                else:
                    avg_params[k] += local_model_params[k] * w
    elif args.federated_optimizer == "FedAvg_seq":
        for k in range(0, len(avg_params)):
            for i in range(0, len(raw_grad_list)):
                local_sample_number, local_model_params = raw_grad_list[i]
                if i == 0:
                    avg_params[k] = local_model_params[k]
                else:
                    avg_params[k] += local_model_params[k]
    elif args.federated_optimizer == "FedOpt":
        pass

    return avg_params


def jax_aggregator(args, raw_grad_list, training_num):
    (num0, avg_params) = raw_grad_list[0]

    if args.federated_optimizer == "FedAvg":
        for k in avg_params.keys():
            for i in range(0, len(raw_grad_list)):
                local_sample_number, local_model_params = raw_grad_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    avg_params[k]["w"] = local_model_params[k]["w"] * w
                    avg_params[k]["b"] = local_model_params[k]["b"] * w
                else:
                    avg_params[k]["w"] += local_model_params[k]["w"] * w
                    avg_params[k]["b"] += local_model_params[k]["b"] * w
    elif args.federated_optimizer == "FedAvg_seq":
        for k in avg_params.keys():
            for i in range(0, len(raw_grad_list)):
                local_sample_number, local_model_params = raw_grad_list[i]
                if i == 0:
                    avg_params[k]["b"] = local_model_params[k]["b"]
                    avg_params[k]["w"] = local_model_params[k]["w"]
                else:
                    avg_params[k]["b"] += local_model_params[k]["b"]
                    avg_params[k]["w"] += local_model_params[k]["w"]
    elif args.federated_optimizer == "FedOpt":
        pass

    return avg_params


def model_aggregator(args, raw_grad_list, training_num):
    if hasattr(args, "ml_engine"):
        if args.ml_engine == "tf":
            return tf_aggregator(args, raw_grad_list, training_num)
        elif args.ml_engine == "jax":
            return jax_aggregator(args, raw_grad_list, training_num)
        else:
            return torch_aggregator(args, raw_grad_list, training_num)
    else:
        return torch_aggregator(args, raw_grad_list, training_num)
