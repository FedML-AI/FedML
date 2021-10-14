echo "------------nvidia-smi------------"
nvidia-smi
echo "------------python3 --version------------"
python3 --version
echo "------------nvcc --version------------"
nvcc --version
echo "------------python3 -c import torch; print(torch.__version__)------------"
python3 -c "import torch; print(torch.__version__)"
echo "------------python3 -c import torch;print(torch.cuda.nccl.version())------------"
python3 -c "import torch;print(torch.cuda.nccl.version())"

# install package here
sudo -H pip3 install setproctitle

sudo -H pip3 install --upgrade wandb
