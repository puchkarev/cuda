# cuda
Learning to use cuda

## Setup

### Install NVIDIA driver and toolkit

1. Install NVIDIA Driver for GPU Support from [https://www.nvidia.com/Download/index.aspx]

2. Install wsl
```
wsl.exe --install
wsl.exe --update
```

3. Enter the wsl environment
```
wsl
```

4. Install some pre-requisites
```
sudo apt update
sudo apt install gcc
sudo apt install g++
sudo apt install cmake
sudo apt-install build-essential
```

5. Replace the GPG key
```
sudo apt-key del 7fa2af80
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$(lsb_release -cs)/x86_64/7fa2af80.pub
```

6. Install Linux x86 CUDA Toolkit

```
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda-repo-wsl-ubuntu-12-6-local_12.6.2-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-6-local_12.6.2-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

7. Inform system of cuda library path:

```
cat << EOF >> ~/.bashrc
# Installed library paths for cuda
export PATH="/usr/local/cuda-12.6/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH"
EOF
source ~/.bashrc
```

## Building the project

1. Fetch this repo:
```
git clone https://github.com/puchkarev/cuda
```

2. Configure and build

```
cd cuda
cd build
cmake ..
make
```

## Runnign the project

To actually run this:

```
./prime --use_cuda=1 --block_size=256 --threads=100 --start=1000 --size=5000
```

There are 3 ways to run this:
1) Single threaded (use_cuda is not set or set to 0 and threads not set or set to 1 or 0)
2) Multi threaded (use_cuda is not set or set to 0 and threads set to >= 2)
3) On GPU using cuda (use_cuda is set to 1 and block_size set to something reasonable, threads is ignored in this case)

The start and size flags indicate the starting value at which to check prime numbers and how many to check.
