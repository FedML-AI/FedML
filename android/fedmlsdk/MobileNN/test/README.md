This Module automized the process of Model Conversion and Performance Comparison.
Specifically, it automized the following things:
1. Convert Pytorch Model to ONNX Model
2. Convert ONNX Model to MNN Model
3. Train MNN Model using cpp fedmlTrainer
4. Train Model using original Pytorch Script
5. Compare cpp and python Result

Prerequisite:
```bash
pip install onnx
pip install mnn

# The MNN lib is required for cpp comilation
cd android/fedmlsdk/MobileNN
git clone https://github.com/FedML-AI/MNN
```
```bash
# Build and prepare the fedmlTrainer
cd ./build/MNN/ 
sh build_x86_linux.sh --debug
cp ./build_x86_linux_debug/main_mnn_train.out ../../test/src/cpp/
```
Configuration:
```python
# main.py
test_data = {
        "dataset_type": "tabular",                              # Model Name
        "training_data_location": "./data/tabular",             # Training Data Location
        "torch_mnn_transformer_path": "./src/python/torch_mnn_transformer.py",  # Pytorch to MNN Model Converter
        "torch_trainer": "./src/python/torch_trainer.py",       # Pytorch Trainer
        "cpp_executable_path": "./src/cpp/main_mnn_train.out",  # Cpp Trainer
}
```
```bash
python main.py
```