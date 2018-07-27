FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update

RUN apt-get install -y --no-install-recommends \
    apt-utils \
    build-essential \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    ca-certificates \
    libjpeg-dev \
    wget \
    software-properties-common

RUN rm -rf /var/lib/apt/lists/* \
    /etc/apt/sources.list.d/cuda.list \
    /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Install tini
ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# Install Python dependencies
RUN conda install -c mutirri demjson --yes
RUN pip install pyyaml tensorboardX 

# Install pytorch
RUN conda install pytorch=0.3.0 torchvision cuda90 -c pytorch --yes

# Install Notebook
RUN pip --no-cache-dir install jupyter

# Config & Cleanup
RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

#Expose port for accessing jupyter notebooks
EXPOSE 8888 8888 

# Make Pythia workspace
RUN mkdir Pythia 

# Add local pythia clone inside container
ADD ./ /Pythia

# Make Pythia current workign dir
WORKDIR "/Pythia"

ENTRYPOINT jupyter notebook --allow-root --ip=127.0.0.1 --no-browser
