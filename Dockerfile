FROM        ubuntu:20.04
MAINTAINER  Paul D. Hein <pauldhein@email.arizona.edu>
CMD         bash

RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get -y install \
      apt-utils \
      build-essential \
      pkg-config \
      curl \
      git \
      tar \
      wget \
      python3-pip \
      doxygen \
      openjdk-8-jdk \
      libgraphviz-dev \
      libsqlite3-dev \
      libboost-all-dev \
      libeigen3-dev \
      libspdlog-dev \
      pybind11-dev \
      libfmt-dev

# Setup the correct version of Python and install/update pip
RUN echo  'alias python=python3' >> ~/.bashrc

WORKDIR /data
# Download SQLite3 database containing model parameterization data.
RUN curl -O http://vanga.sista.arizona.edu/delphi_data/delphi.db
RUN curl -O http://vanga.sista.arizona.edu/delphi_data/model_files.tar.gz

# Set the environment variable DELPHI_DB to point to the SQLite3 database.
ENV DELPHI_DB=/data/delphi.db
ENV MODEL_FILES=/data/source_model_files
ENV AUTOMATES_LOC=/fake/path/for/now
ENV EMB_LOC=/fake/path/for/now

# Build the delphi testing environment
WORKDIR /repo
RUN git clone https://github.com/ml4ai/delphi.git

WORKDIR /repo/delphi
RUN pip3 install cython futures
