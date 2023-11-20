FROM nvidia/cuda:12.3.0-base-ubuntu20.04

# Do all tasks to be done with root privileged
RUN  apt update && \
     apt install -y wget git nano && \
     apt-get clean && rm -rf /var/lib/apt/lists/*

# Build arguments
ARG LDAP_USERNAME
ARG LDAP_GROUPNAME
ARG LDAP_UID
ARG LDAP_GID

# Create local user and group
RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID}
RUN useradd -m -U -s /bin/bash -G ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}

## Switch to the local user
USER ${LDAP_USERNAME}
WORKDIR /home/${LDAP_USERNAME}

# Additional configuration for the local user

## Install conda in your home
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh  \
    && bash ~/miniconda.sh -b -p $HOME/miniconda \
    && rm ~/miniconda.sh

ENV PATH /home/${LDAP_USERNAME}/miniconda/bin:${PATH}

## Create a conda env with required packages
RUN conda install mamba -n base -c conda-forge -y
RUN mamba create -n venv python=3.8 -y
COPY environment.yml /tmp/environment.yml
RUN mamba env update -n venv -f /tmp/environment.yml
RUN echo "source activate venv" > ~/.bashrc

ENV PATH /home/${LDAP_USERNAME}/.conda/envs/venv/bin:$PATH