# install cuda toolkit 11.7
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt -y install cuda-toolkit-11-7

# incase of GPG error
curl -sL "http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xF60F4B3D7FA2AF80" | sudo apt-key add -

# setup cuda 11.7 path
export PATH=/usr/local/cuda-11.7/bin:$PATH


# Verify if cuda toolkit 11.7 is avaiable
nvcc --version

# Get no of GPUS
nvidia-smi

# install mpi specific libs
sudo apt install libopenmpi-dev
# install torch with cuda 11.7
pip install torch==1.13.1+cu118 torchvision==0.14.1+cu118 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# run training
deepspeed --num_gpus=2 train.py --config config.json

# to check GPU usage
watch -n1 nvidia-smi