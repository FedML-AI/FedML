import os
import subprocess
import difflib
def execute_cpp(cpp_executable_path, cpp_model_name, mnn_model_path, training_data_location):
    command = f'{cpp_executable_path} {cpp_model_name} {mnn_model_path} {training_data_location}'
    print(f"command: {command}")

    if not os.path.exists("./runtime_result"):
        os.makedirs("./runtime_result")

    cpp_training_result_file_path = "./runtime_result/" + cpp_model_name + "_cpp.txt"
    with open(cpp_training_result_file_path, 'w') as f:
        subprocess.run(command, shell=True, stdout=f, text=True)

def transform_torch_model_mnn(transformer_loc, mnn_model_direction, model_name):
    command = f'python {transformer_loc} {mnn_model_direction} {model_name}'
    print(f"command: {command}")
    # this will generate tabular.mnn, tabular.onnx, tabular.pth
    subprocess.run(command, shell=True)

def execute_torch(torch_trainer, train_data_path, initial_weights_path, model_name):
    command = f'python {torch_trainer} {train_data_path} {initial_weights_path} {model_name}'
    print(f"command: {command}")

    py_training_result_file_path = "./runtime_result/" + model_name + "_py.txt"
    with open(py_training_result_file_path, 'w') as f:
        subprocess.run(command, shell=True, stdout=f, text=True)

def diff_cpp_python(model_name):
    with open("./runtime_result/" + model_name + "_cpp.txt", 'r') as f:
        cpp_result = f.readlines()
    with open("./runtime_result/" + model_name + "_py.txt", 'r') as f:
        python_result = f.readlines()
    diff = difflib.ndiff(cpp_result, python_result)
    with open("./runtime_result/" + model_name + "_diff.txt", 'w') as f:
        f.writelines(diff)