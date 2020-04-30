FROM        ubuntu:19.10
MAINTAINER  Terrence Lim <hlim1@email.arizona.edu>
CMD         bash

# ==============================================================================
# INSTALL SOFTWARE VIA THE UBUNTU PACKAGE MANAGER
# ==============================================================================
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get -y --no-install-recommends install \
  git \
  # build-essential \
  # gcc \
  # pkg-config \
  python3 \
  python3-dev \
  python3-pip \
  default-jdk

RUN git config --global user.email "hlim1@email.arizona.edu"
RUN git config --global user.name "Terrence Lim"
# ==============================================================================

# ==============================================================================
# INSTALL PYTHON PACKAGES VIA PIP
# NOTE: Packages are installed in stages to avoid memory errors on DockerHub
# ==============================================================================
# Upgrading setuptools and installing wheel
# RUN pip3 install --upgrade setuptools
# RUN pip3 install wheel

# Scientific computing packages
RUN pip3 install networkx==2.4 numpy==1.18.2
# ==============================================================================

# ==============================================================================
# SETUP THE AUTOMATES REPOSITORY AND ENVIRONMENT
# ==============================================================================
RUN git clone https://github.com/ml4ai/delphi.git \
    && cd delphi \
    && git checkout dssat_pet

ENV PYTHONPATH="/delphi:$PYTHONPATH"
ENV DELPHI_DB=/delphi/data/delphi.db

WORKDIR /delphi

# Execute f2grfn_standalone.py script.
# -f <fortran_source_file>
# -r <root_directory>
CMD ["python3.7", "scripts/f2grfn_standalone.py", "-f" ,"/delphi/tests/data/program_analysis/PETPT.for", "-r", "/delphi"]

# ==============================================================================
