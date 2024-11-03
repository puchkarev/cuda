# cuda
Learning to use cuda

Things I did for setup:

```
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$(lsb_release -cs)/x86_64/7fa2af80.pub
```

Added to ~/.bashrc:

```
# Installed library paths for cuda
export PATH="/usr/local/cuda-12.6/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH"
```

To actually build this:

```
git clone https://github.com/puchkarev/cuda
cd cuda
cd build
cmake ..
make
```

To actually run this:

```
./prime --use_cuda=1 --block_size=256 --threads=100 --start=1000 --size=5000
```

There are 3 ways to run this:
1) Single threaded (use_cuda is not set or set to 0 and threads not set or set to 1 or 0)
2) Multi threaded (use_cuda is not set or set to 0 and threads set to >= 2)
3) On GPU using cuda (use_cuda is set to 1 and block_size set to something reasonable, threads is ignored in this case)

The start and size flags indicate the starting value at which to check prime numbers and how many to check.
