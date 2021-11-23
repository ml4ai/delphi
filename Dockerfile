FROM        ubuntu:20.04
CMD         bash

ENV DEBIAN_FRONTEND noninteractive
# Set up virtual environment
ENV VIRTUAL_ENV=/venv
# Set the environment variable DELPHI_DB to point to the SQLite3 database.
ENV DELPHI_DB=/delphi/data/delphi.db
ENV MODEL_FILES=/delphi/data/source_model_files

RUN apt-get update \
    && apt-get -y --no-install-recommends install \
      ca-certificates \
      build-essential \
      libboost-all-dev \
      pkg-config \
      cmake \
      curl \
      git \
      tar \
      wget \
      doxygen \
      graphviz \
      libgraphviz-dev \
      libsqlite3-dev \
      libeigen3-dev \
      libfmt-dev \
      librange-v3-dev \
      nlohmann-json3-dev

# build served from source
RUN curl -LO https://github.com/meltwater/served/archive/refs/tags/v1.6.0.tar.gz \
    && tar -xzf v1.6.0.tar.gz \
    && cd served-1.6.0 \
    &&       mkdir build \
    &&       cd build \
    &&       cmake .. \
    &&       make -j `nproc` install \
    && cd ..

COPY . /delphi
WORKDIR /delphi

# build delphi_rest_api
RUN mkdir -p data \ 
  && mkdir build \
  && cd build \
  && cmake .. -DBUILD_PYTHON_BINDINGS=OFF \
  && make -j `nproc` delphi_rest_api \
  && cd ..
