echo "Hello, Here is the launch platform."
echo "Current directory is as follows."
pwd
# sleep 15
echo "Current GPU information is as follows."
nvidia-smi # Print GPU information
gpustat
echo "Download the file from http://212.183.159.230/200MB.zip ..."
wget http://212.183.159.230/200MB.zip
rm ./200MB.zip*
echo "The downloading task has finished."
# echo "Training the vision transformer model using PyTorch..."
# python vision_transformer.py --epochs 1
