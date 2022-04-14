import MNN
import torch
import copy

F = MNN.expr


def transform_mnn_to_tensor(mnn_path):
    model_path = mnn_path
    var_map = F.load_as_dict(model_path)
    input_dicts, output_dicts = F.get_inputs_and_outputs(var_map)
    input_names = [n for n in input_dicts.keys()]
    output_names = [n for n in output_dicts.keys()]
    input_vars = [input_dicts[n] for n in input_names]
    output_vars = [output_dicts[n] for n in output_names]
    module = MNN.nn.load_module(input_vars, output_vars, False)

    tensor_params_dict = {}
    for idx_layer in range(len(module.parameters)):
        module.parameters[idx_layer].fix_as_const()
        mnn_layer_weights_np_arr = copy.deepcopy(module.parameters[idx_layer].read())
        tensor_params_dict[idx_layer] = torch.from_numpy(mnn_layer_weights_np_arr).detach()

    return tensor_params_dict

def transform_tensor_to_mnn(mnn_path, tensor_dict, mnn_save_path):
    model_path = mnn_path
    var_map = F.load_as_dict(model_path)
    input_dicts, output_dicts = F.get_inputs_and_outputs(var_map)
    input_names = [n for n in input_dicts.keys()]
    output_names = [n for n in output_dicts.keys()]
    input_vars = [input_dicts[n] for n in input_names]
    output_vars = [output_dicts[n] for n in output_names]
    module = MNN.nn.load_module(input_vars, output_vars, False)
    input_shape = F.shape(input_vars[0])


    mnn_params_list = []
    for idx_layer in range(len(tensor_dict)):
        pt_layer_weights_np_arr = tensor_dict[idx_layer].numpy()
        tmp = F.const(pt_layer_weights_np_arr, list(pt_layer_weights_np_arr.shape))
        tmp.fix_as_trainable()
        mnn_params_list.append(tmp)

    module.load_parameters(mnn_params_list)
    predict = module.forward(F.placeholder(input_shape.read(), F.NCHW))
    F.save([predict], mnn_save_path)


if __name__ == "__main__":
    mnn_path = "../model_test/lenet_cifar10.mnn"
    save_path = "../model_test/agg.mnn"
    save_an_path = "../model_test/test.mnn"
    a = transform_mnn_to_tensor(mnn_path)
    print(a[3])
    transform_tensor_to_mnn(mnn_path, a, save_path)

    c = transform_mnn_to_tensor(save_path)
    print(c[3])

    transform_tensor_to_mnn(mnn_path, c, save_an_path)

    b = transform_mnn_to_tensor(save_an_path)
    print(b[3])




