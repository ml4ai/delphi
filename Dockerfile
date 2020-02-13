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
      cmake \
      time \
      curl \
      git \
      tar \
      wget \
      python3-pip \
      doxygen \
      openjdk-8-jdk \
      libgraphviz-dev \
      nlohmann-json3-dev \
      libsqlite3-dev \
      libboost-all-dev \
      libeigen3-dev \
      libspdlog-dev \
      pybind11-dev \
      libfmt-dev \
      librange-v3-dev \
    && pip3 install cython futures \
    && echo 'alias python=python3' >> ~/.bashrc \
    && git clone https://github.com/ml4ai/delphi \
    && curl http://vanga.sista.arizona.edu/delphi_data/delphi.db -o delphi/data/delphi.db \
    && curl http://vanga.sista.arizona.edu/delphi_data/model_files.tar.gz -o delphi/data/model_files.tar.gz


# Set the environment variable DELPHI_DB to point to the SQLite3 database.
RUN apt-get install -y python3-venv && python3 -m venv delphi_venv \
    && . delphi_venv/bin/activate
WORKDIR /delphi
RUN pip install -e .[test,docs,aske]
ENV DELPHI_DB=/delphi/data/delphi.db
ENV MODEL_FILES=/delphi/data/source_model_files

# Build the delphi testing environment
