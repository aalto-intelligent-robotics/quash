FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_HOME=$CUDA_HOME
# tested to work with 3.9, 3.10
ARG PYTHON_VERSION=3.10

ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# install python
RUN apt update
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN rm -f /usr/bin/python3
RUN rm -f /usr/bin/python
RUN apt install -y python${PYTHON_VERSION}
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python3
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN apt install -y libgl-dev
RUN apt install -y libglib2.0-0
RUN apt install -y libgl1
RUN apt install -y libgl1-mesa-glx
RUN apt install -y libnss3
RUN apt install -y libsm6
RUN apt install -y libxext6
RUN apt install -y libxrender-dev
RUN apt install -y pylint
RUN apt install -y mypy
RUN apt install -y libqt5gui5
RUN apt install -y python3-dev
RUN apt install -y libpython${PYTHON_VERSION}-dev
RUN apt install -y python3-pip
RUN apt install -y python${PYTHON_VERSION}-tk
RUN apt install -y python${PYTHON_VERSION}-distutils

# install important utils
RUN apt install -y git
RUN apt install -y screen
RUN apt install -y tmux
RUN apt install -y zip
RUN apt install -y unzip
RUN apt install -y vim
RUN apt install -y curl
RUN apt install -y wget

# install clip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}
RUN python${PYTHON_VERSION} -m pip --version
# RUN python -m pip install html5lib
# RUN apt install -y python3-html5lib
RUN pip3 install --upgrade pip
RUN pip3 install ftfy regex tqdm
RUN pip3 install git+https://github.com/openai/CLIP.git

# install vlmap dependecies
RUN pip3 install opencv-python-headless
RUN pip3 install Pillow
RUN pip3 install pandas
RUN pip3 install h5py
RUN pip3 install matplotlib
RUN pip3 install scipy
RUN pip3 install timm
RUN pip3 install pytorch-lightning
RUN pip3 install transformers
RUN pip3 install seaborn
RUN pip3 install scikit-learn
RUN pip3 install pycocotools
RUN pip3 install black
RUN pip3 install tzlocal
RUN pip3 install openai
RUN pip3 install datasets
RUN pip3 install pandasgui
RUN pip3 install shapely
RUN pip3 install optuna
RUN pip3 install gdown
RUN pip3 install tensorflow==2.17.1

# other stuff
RUN pip3 install wonderwords

# LSEG
# Note: This has to be manually installed when running - idk why
#RUN pip3 install git+https://github.com/zhanghang1989/PyTorch-Encoding/
#RUN pip3 install torch-encoding
RUN pip3 install tensorboardX
RUN pip3 install ninja

# BLIP-2
RUN pip install git+https://github.com/huggingface/transformers
RUN pip install accelerate>=0.26.0
RUN pip install bitsandbytes
RUN pip install -U --ignore-installed psutil

ENV QT_DEBUG_PLUGINS=0

# fix pandasgui
RUN sed -i "s/SHORTCUT_PATH/#SHORTCUT_PATH/" /usr/local/lib/python${PYTHON_VERSION}/dist-packages/pandasgui/constants.py

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
WORKDIR /home/user/<path/to/quash>

ENV TF_CPP_MIN_LOG_LEVEL=3
# If problems with VSCode dev container, this helped
# export QT_QPA_PLATFORM=offscreen
