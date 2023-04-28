import logging

import torch

from .torch_process_group_manager import TorchProcessGroupManager
from ...core.common.ml_engine_backend import MLEngineBackend


def convert_numpy_to_torch_data_format(args, batched_x, batched_y):
    import torch
    import numpy as np

    if args.model == "cnn":
        batched_x = torch.from_numpy(np.asarray(batched_x)).float().reshape(-1, 28, 28)  # CNN_MINST
    else:
        batched_x = torch.from_numpy(np.asarray(batched_x)).float()  # LR_MINST or other

    batched_y = torch.from_numpy(np.asarray(batched_y)).long()
    return batched_x, batched_y


def convert_numpy_to_tf_data_format(args, batched_x, batched_y):
    # https://www.tensorflow.org/api_docs/python/tf/convert_to_tensor
    import tensorflow as tf
    import numpy as np

    if args.model == "cnn":
        batched_x = tf.convert_to_tensor(np.asarray(batched_x), dtype=tf.float32)  # CNN_MINST
        batched_x = tf.reshape(batched_x, [-1, 28, 28])
    else:
        batched_x = tf.convert_to_tensor(np.asarray(batched_x), dtype=tf.float32)  # LR_MINST or other

    batched_y = tf.convert_to_tensor(np.asarray(batched_y), dtype=tf.int64)
    return batched_x, batched_y


def convert_numpy_to_jax_data_format(args, batched_x, batched_y):
    import numpy as np

    if args.model == "cnn":
        batched_x = np.asarray(batched_x, dtype=np.float32)  # CNN_MINST
        batched_x = np.reshape(batched_x, [-1, 28, 28])
    else:
        batched_x = np.asarray(batched_x, dtype=np.float32)  # LR_MINST or other

    batched_y = np.asarray(batched_y, dtype=np.float32)
    return batched_x, batched_y


def convert_numpy_to_mxnet_data_format(args, batched_x, batched_y):
    from mxnet import np as mx_np

    if args.model == "cnn":
        batched_x = mx_np.array(batched_x)
        batched_x = mx_np.reshape(batched_x, [-1, 28, 28])  # pylint: disable=E1101
    else:
        batched_x = mx_np.array(batched_x)

    batched_y = mx_np.array(batched_y)
    return batched_x, batched_y


def convert_numpy_to_ml_engine_data_format(args, batched_x, batched_y):
    if hasattr(args, MLEngineBackend.ml_engine_args_flag):
        if args.ml_engine == MLEngineBackend.ml_engine_backend_tf:
            return convert_numpy_to_tf_data_format(args, batched_x, batched_y)
        elif args.ml_engine == MLEngineBackend.ml_engine_backend_jax:
            return convert_numpy_to_jax_data_format(args, batched_x, batched_y)
        elif args.ml_engine == MLEngineBackend.ml_engine_backend_mxnet:
            return convert_numpy_to_mxnet_data_format(args, batched_x, batched_y)
        else:
            return convert_numpy_to_torch_data_format(args, batched_x, batched_y)
    else:
        return convert_numpy_to_torch_data_format(args, batched_x, batched_y)


def is_torch_device_available(args, device_type):
    if device_type == MLEngineBackend.ml_device_type_gpu:
        if torch.cuda.is_available():
            return True
        return False
    elif device_type == MLEngineBackend.ml_device_type_mps:
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
    elif device_type == MLEngineBackend.ml_device_type_gpu:
        return True

    return False


def is_mxnet_device_available(args, device_type):
    if device_type == MLEngineBackend.ml_device_type_cpu:
        return True
    elif device_type == MLEngineBackend.ml_device_type_gpu:
        try:
            import mxnet as mx

            gpus = mx.device.num_gpus()  # pylint: disable=E1101
        except Exception as ex:
            return False

        if gpus > 0:
            return True

    return False


def is_device_available(args, device_type=MLEngineBackend.ml_device_type_gpu):
    if hasattr(args, MLEngineBackend.ml_engine_args_flag):
        if args.ml_engine == MLEngineBackend.ml_engine_backend_tf:
            import tensorflow as tf

            devices = tf.config.list_physical_devices(device_type.upper())
            if len(devices) > 0:
                return True
            return False
        elif args.ml_engine == MLEngineBackend.ml_engine_backend_jax:
            try:
                import jax

                device_count = jax.device_count(device_type)
                if device_count > 0:
                    return True
            except Exception as ex:
                return False

            return False
        elif args.ml_engine == MLEngineBackend.ml_engine_backend_mxnet:
            return is_mxnet_device_available(args, device_type)
        else:
            return is_torch_device_available(args, device_type)
    else:
        return is_torch_device_available(args, device_type)


def get_torch_device(args, using_gpu, device_id, device_type):
    logging.info(
        "args = {}, using_gpu = {}, device_id = {}, device_type = {}".format(args, using_gpu, device_id, device_type)
    )
    if using_gpu:
        gpu_id = device_id if device_id is not None else args.local_rank

        if torch.cuda.is_available() and device_type == MLEngineBackend.ml_device_type_gpu:
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(int(gpu_id))
        elif device_type == MLEngineBackend.ml_device_type_mps:
            # https://pytorch.org/docs/master/notes/mps.html
            device = torch.device(MLEngineBackend.ml_device_type_mps)
        else:
            device = torch.device(MLEngineBackend.ml_device_type_cpu)

        return device
    else:
        return torch.device(MLEngineBackend.ml_device_type_cpu)


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


def get_mxnet_device(args, using_gpu, device_id, device_type):
    import mxnet as mx

    if using_gpu:
        return mx.gpu(device_id)
    else:
        return mx.cpu()


def get_device(args, device_id=None, device_type="cpu"):
    using_gpu = True if (hasattr(args, "using_gpu") and args.using_gpu is True) else False

    if hasattr(args, MLEngineBackend.ml_engine_args_flag):
        if args.ml_engine == MLEngineBackend.ml_engine_backend_tf:
            return get_tf_device(args, using_gpu, device_id, device_type)
        elif args.ml_engine == MLEngineBackend.ml_engine_backend_jax:
            return get_jax_device(args, using_gpu, device_id, device_type)
        elif args.ml_engine == MLEngineBackend.ml_engine_backend_mxnet:
            return get_mxnet_device(args, using_gpu, device_id, device_type)
        else:
            return get_torch_device(args, using_gpu, device_id, device_type)
    else:
        return get_torch_device(args, using_gpu, device_id, device_type)


def dict_to_device(args, dict_obj, device):
    if hasattr(args, MLEngineBackend.ml_engine_args_flag):
        if args.ml_engine == MLEngineBackend.ml_engine_backend_tf:
            with device:
                dict_ret = dict_obj
                return dict_ret
        elif args.ml_engine == MLEngineBackend.ml_engine_backend_jax:
            import jax

            return jax.device_put(dict_obj, device)
        elif args.ml_engine == MLEngineBackend.ml_engine_backend_mxnet:
            with device:
                dict_ret = dict_obj
                return dict_ret
        else:
            return dict_obj.to(device)
    else:
        return dict_obj.to(device)


def model_params_to_device(args, params_obj, device):
    if hasattr(args, MLEngineBackend.ml_engine_args_flag):
        if args.ml_engine == MLEngineBackend.ml_engine_backend_tf:
            with device:
                params_ret = params_obj
                return params_ret
        elif args.ml_engine == MLEngineBackend.ml_engine_backend_jax:
            for key in params_obj.keys():
                params_obj[key] = dict_to_device(args, params_obj[key], device)
        elif args.ml_engine == MLEngineBackend.ml_engine_backend_mxnet:
            with device:
                params_ret = params_obj
                return params_ret
        else:
            for key in params_obj.keys():
                params_obj[key] = dict_to_device(args, params_obj[key], device)
    else:
        for key in params_obj.keys():
            params_obj[key] = dict_to_device(args, params_obj[key], device)

    return params_obj


def model_to_device(args, model_obj, device):
    if hasattr(args, MLEngineBackend.ml_engine_args_flag):
        if args.ml_engine == MLEngineBackend.ml_engine_backend_tf:
            with device:
                model_ret = model_obj
                return model_ret
        elif args.ml_engine == MLEngineBackend.ml_engine_backend_jax:
            return model_obj
        elif args.ml_engine == MLEngineBackend.ml_engine_backend_mxnet:
            model_obj.reset_device(device)
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


# Todo: add tf ddp
def tf_model_ddp(args, model_obj, device):
    process_group_manager, model = None, model_obj
    return process_group_manager, model


# Todo: add jax ddp
def jax_model_ddp(args, model_obj, device):
    process_group_manager, model = None, model_obj
    return process_group_manager, model


# Todo: add mxnet ddp
def mxnet_model_ddp(args, model_obj, device):
    process_group_manager, model = None, model_obj
    return process_group_manager, model


def model_ddp(args, model_obj, device):
    process_group_manager, model = None, model_obj
    if hasattr(args, MLEngineBackend.ml_engine_args_flag):
        if args.ml_engine == MLEngineBackend.ml_engine_backend_tf:
            return tf_model_ddp(args, model_obj, device)
        elif args.ml_engine == MLEngineBackend.ml_engine_backend_jax:
            return jax_model_ddp(args, model_obj, device)
        elif args.ml_engine == MLEngineBackend.ml_engine_backend_mxnet:
            return mxnet_model_ddp(args, model_obj, device)
        else:
            process_group_manager, model = torch_model_ddp(args, model_obj, device)
    else:
        process_group_manager, model = torch_model_ddp(args, model_obj, device)

    return process_group_manager, model

