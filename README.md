# Keras OCR Flask Service

This service, built with Python and Flask, utilizes Optical Character Recognition (OCR) technology to pinpoint words in images with remarkable precision. Simply submit an image along with a target word via a POST request, and the tool goes to work. If the word is located, it returns the exact coordinates of where it was found within the image. If the word remains elusive, the service will let you know it couldn't find it. 

- [Keras OCR Flask Service](#keras-ocr-flask-service)
- [API](#api)
- [Running with Docker Compose](#running-with-docker-compose)
  - [1 | Install Docker](#1--install-docker)
  - [2 | Install NVIDIA Driver](#2--install-nvidia-driver)
  - [3 | Install NVIDIA Container Toolkit](#3--install-nvidia-container-toolkit)
  - [4 | Run with Docker Compose](#4--run-with-docker-compose)
- [Devlopment docker container](#devlopment-docker-container)
  - [1 | Build the image](#1--build-the-image)
  - [2 | Run the container](#2--run-the-container)
  - [3 | Install dependencies and run](#3--install-dependencies-and-run)
- [Running natively](#running-natively)
  - [1 | Install Miniconda](#1--install-miniconda)
  - [2 | Create a conda environment](#2--create-a-conda-environment)
  - [3 | Install GPU Driver, CUDA Toolkit, and cuDNN.](#3--install-gpu-driver-cuda-toolkit-and-cudnn)
  - [4 | Run Keras OCR API](#4--run-keras-ocr-api)

# API
1. Send a post request to /process as form-data
2. Include the screenshot as "file" and the word you are searching for as "word"
3. Will return a json response

```
{
    "result": "found",
    "x": 3464,
    "y": 1872
}

or 

{
    "result": "not found"
}
```


# Running with Docker Compose

For these instructions we use [Ubuntu Desktop 24.04 LTS](https://ubuntu.com/blog/ubuntu-desktop-24-04-noble-numbat-deep-dive)

## 1 | Install Docker

We follow the [official Docker instructions](https://docs.docker.com/engine/install/ubuntu/) for installing on Ubuntu. We implore you to read their instructions to install but here is the steps for brevity's sake.

> Docker desktop for Ubuntu does not work with this project, ensure you are installing Docker Engine and not [Docker Desktop](https://docs.docker.com/desktop/install/linux/ubuntu/).

1. Uninstall old versions

```bash
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done
```

2. Set up Docker's apt repository

```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```

3. Install the docker packages

```bash
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

4. Manage docker as a non-root user

```bash
# add your current user to the docker group
sudo usermod -aG docker $USER
```
> Log in and log out after this step for the new group to take affect.

```bash
# test you can run docker without sudo
docker run hello-world
```

5. (Optional) If docker is not starting for you try this workaround to fix a known [issue with Ubuntu 24.04 and Docker Desktop](https://github.com/docker/desktop-linux/issues/209#issuecomment-2083540338) (which you shouldn't have installed if you followed these instructions!)

```bash
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
```

## 2 | Install NVIDIA Driver

This project will work on a processor but its success in game automation largely depends on the performance boost of accelerating TensorFlow on a CUDA enabled NVIDIA GPU.

1. Check if you already have a supported driver install. Commonly a working driver will be found during Ubuntu's installations if you allowed third party drivers.

```bash
# if you receive output like the image below you have a working driver
nvidia-smi
```

![expected nvidia-smi command output](doc/nvidia_smi_command_output.png)

If no output is given, or if the "command is not found" follow these instructions to install an appropriate driver:

See [CUDA GPUs - Compute Capability](https://developer.nvidia.com/cuda-gpus) to find out which version of CUDA your GPU supports. In this case the 4090 supports 8.9.

> [Ubuntu Nvidia Driver Instructions](https://help.ubuntu.com/community/NvidiaDriversInstallation)

```bash
# use below if you want automatic detection
sudo ubuntu-drivers install

# you can view available drivers and select a specific one to install instead
sudo ubuntu-drivers list
```

## 3 | Install NVIDIA Container Toolkit

Again, like the sections above, we implore you to read the [official NVIDIA Container Toolkit documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and use ours as reference.


1. Add the repository to apt
   
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

2. Update the list

```bash
sudo apt-get update
```

3. Install the packages

```bash
sudo apt-get install -y nvidia-container-toolkit
```

4. Configure docker

```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

5. Restart docker

```bash
sudo systemctl restart docker
```

6. Verify it all worked
   
```bash
sudo docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

This command should have identical output as if you ran `nvidia-smi` on the host system.

## 4 | Run with Docker Compose

1. Open a terminal in the root directory of the repository

2. Build and start the service

```bash
docker compose up
```
3. Stop the service

```bash
docker compose stop
```

4. Restart the service

```bash
docker compose restart
```

5. Stop and remove all containers

```bash
docker compose down
```

# Devlopment docker container

You can run and build the container this way if you want to be able to make changes to the code and pick those changes up without having to rebuild the whole container.

## 1 | Build the image
```bash
docker build -t <tag-name> -f Dockerfile.dev . 
```

## 2 | Run the container
Ensure you run this command from the root of the respository so the correct directory is mounted to the container.

```bash
sudo docker run --shm-size=1g --ulimit memlock=-1 --name keras-ocr -it -v $(pwd):/repo --gpus all <tag-name>
```

## 3 | Install dependencies and run
Once you run the command from step 2 you should have been greeted to an SSH session into the running container.

```bash
# install general dependencies
pip install -r requirements_docker.txt

# install tensorflow
pip install tensorflow==2.12.0

# test if GPU is detected
python test_cudapresence.py

# run the service
python keras_server.py
```

# Running natively

To start we will install TensorFlow for Linux following the [official documentation](https://www.tensorflow.org/install/pip#linux). Our instructions assume you are using an Nvidia graphics card for CUDA acceleration.  

- We standardized on installing Keras OCR on Ubuntu Server 20.04 and these instructions are from a fresh install.
- This project is currently using TensorFlow 2.12.0

## 1 | Install Miniconda

Miniconda is the recommended approach for installing TensorFlow with GPU support, we follow this advice.

1. Execute `curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh`
2. Then execute `bash Miniconda3-latest-Linux-x86_64.sh`
    - You may need to restart your terminal or source ~/.bashrc to enable the conda command.
    - Use conda -V to test if it is installed successfully.


## 2 | Create a conda environment

We will create a conda environment in which to operate. In Labs we use the `/home/<user>` directory. We stick with the home directory because most of our deploys are to native machines with no other services; they are meant just for Keras.

Staring in the home directory of the user (`cd ~`):

1. Execute `conda create --name tf python=3.10`
2. Activate the environment with `conda activate tf` 


## 3 | Install GPU Driver, CUDA Toolkit, and cuDNN.

> You can skip this part if you just want to run Keras on the CPU, however many of the game tests that use Keras will fail as the CPU is not fast enough for some of the timings expected in the game tests.


Use [CUDA GPUs - Compute Capability](https://developer.nvidia.com/cuda-gpus) to find out which version of CUDA your GPU supports. In this case the 4090 supports 8.9.

![Alt text](doc/compute_capability.png)

[TensorFlow tested build configurations](https://www.tensorflow.org/install/source#gpu)

1. [Install the graphics card driver.](https://help.ubuntu.com/community/NvidiaDriversInstallation)
   - Execute `sudo ubuntu-drivers install` if you want automatic detection.
   - If you want to see the available drivers, run `sudo ubuntu-drivers list`
2. Use the following command to verify it is installed `nvidia-smi`.
3. Install Cuda Tool Kit with Conda
   - Execute `conda install -c conda-forge cudatoolkit=11.8.0`
4. Install cuDNN with pip.
   - Execute `pip install nvidia-cudnn-cu11==8.9.5` 
   - `8.6.0.163` for 30's series GPUs.
5. Configure the system paths. You can do it with the following command every time you start a new terminal after activating your conda environment.
    - `CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))`
    - `export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH`
6. For your convenience it is recommended that you automate it with the following commands. The system paths will be automatically configured when you activate this conda environment.
    - `mkdir -p $CONDA_PREFIX/etc/conda/activate.d`
    - `echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh`
    - `echo 'export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh`
7. Install TensorFlow
   1. `pip install --upgrade pip`
   2. `pip install tensorflow==2.12.0`

Test it works on CPU
```
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

Test it works on GPU
```
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 4 | Run Keras OCR API
Now we can install the rest of the dependencies and test if our API is working.

1. Install the rest of the dependencies.
    - Execute `pip install -r requirements.txt`
2. Test Keras and Tensorflow.
    - Execute `python3 test_cudapresence.py`
    - It should print out that GPU is available.
3. Execute `run-keras-service.sh`
