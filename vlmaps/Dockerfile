FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ARG CUDA_HOME=$CUDA_HOME
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# install python
RUN apt update
RUN apt install -y python3.8
RUN ln -s /usr/bin/python3.8 /usr/bin/python3
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN apt install -y libgl-dev
RUN apt install -y libglib2.0-0
RUN apt install -y libgl1
RUN apt install -y libgl1-mesa-glx

RUN apt install -y python3-pip
RUN pip3 install --upgrade pip

# install important utils
RUN apt install -y git

# install clip
RUN pip3 install ftfy regex tqdm
RUN pip3 install git+https://github.com/openai/CLIP.git

# install vlmap dependecies
RUN pip3 install opencv-python
RUN pip3 install Pillow
RUN pip3 install pandas
RUN pip3 install h5py
RUN pip3 install matplotlib
RUN pip3 install scipy
RUN pip3 install torchvision==0.12.0
RUN pip3 install pytorch-lightning==1.0.0
RUN pip3 install timm==0.6.13
RUN pip3 install jupyter
RUN pip3 install transformers==4.28.0
RUN pip3 install scikit-learn
RUN pip3 install argparse
RUN pip3 install pathlib
RUN pip3 install tqdm
RUN pip3 install fitter
RUN pip3 install progressbar
RUN pip3 install pykeops
RUN pip3 install geomloss
# RUN pip3 install pandasgui
RUN pip3 install joblib
RUN pip3 install seaborn
RUN pip3 install pycocotools
RUN pip3 install torch-encoding==1.2.1
RUN pip3 install opencv-python==4.1.2.30
RUN pip3 install imageio==2.23.0
RUN pip3 install imageio-ffmpeg==0.4.7
RUN pip3 install altair==4.1.0
RUN pip3 install timm==0.6.12
RUN pip3 install tensorboardX==2.2
RUN pip3 install matplotlib==3.6.2
RUN pip3 install test-tube==0.7.5
RUN pip3 install wandb==0.13.7
RUN pip3 install h5py==3.7.0
RUN pip3 install cmake
RUN pip3 install open3d==0.17.0
RUN pip3 install grad-cam==1.4.6
RUN pip3 install networkx==2.8.8
RUN pip3 install mpl_point_clicker==0.3.1
RUN pip3 install ai2thor==5.0.0
RUN pip3 install rospkg==1.4.0
RUN pip3 install pyvisgraph==0.2.1
RUN pip3 install gdown==4.6.0
RUN pip3 install hydra-core==1.3.1
RUN pip3 install google-cloud-speech==2.17.0
RUN pip3 install sentence_transformers==2.2.2
RUN pip3 install shapely==2.0.2
RUN pip3 install scikit-image
RUN pip3 install hydra-core
RUN pip3 install notebook
RUN pip3 install openai
RUN pip3 install hydra-core
RUN pip3 install torchtext==0.6
RUN pip3 install tensorflow[and-cuda]==2.11.0
RUN pip3 install numba
RUN pip3 install wonderwords
RUN pip3 install diptest
RUN pip3 install kneed
RUN pip3 install optuna

# open cv requirements
RUN apt install -y libsm6
RUN apt install -y libxrender-dev
RUN apt install -y python3-tk

# install utils
RUN apt install -y screen

# Needed for unzipping downloaded VLMaps dataset
RUN apt install -y zip unzip

RUN apt install -y libopenexr-dev
RUN pip3 install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# vscode run
RUN apt install -y libfontconfig
RUN apt install -y libxcb-*
RUN apt install -y libxkbcommon-x11-0
RUN apt install -y libdbus-1-3
RUN apt install -y libosmesa6
RUN apt install -y libosmesa6-dev
RUN apt install -y x11-apps
RUN apt install bc

# vscode extensions
RUN pip install mypy
RUN pip install pylint

#RUN pip install -e /home/user/<path/to/quash>

# add user
RUN apt install -y sudo
ARG UID
ARG GID
RUN groupadd -g "${GID}" user
RUN useradd --create-home --no-log-init -u "${UID}" -g "${GID}" user
RUN usermod -a -G sudo user
RUN echo "user:user"|chpasswd
RUN usermod --shell /bin/bash user
USER user

ENV TERM=xterm-256color
#RUN echo "PS1='\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\[\033[01;93m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> /home/user/.bashrc
WORKDIR /home/user/code/vlmaps
ENV TORCH_CUDA_ARCH_LIST=8.6
ENV XDG_RUNTIME_DIR=/run/user/${UID}
ENV TF_CPP_MIN_LOG_LEVEL=3
