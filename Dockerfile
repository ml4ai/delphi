FROM        ubuntu:20.04
CMD         bash

ENV DEBIAN_FRONTEND noninteractive
# Set the environment variable DELPHI_DB to point to the SQLite3 database.
ENV DELPHI_DB=/delphi/data/delphi.db
ENV MODEL_FILES=/delphi/data/source_model_files


RUN apt-get update \
    && apt-get -y --no-install-recommends install \
      build-essential \
      libboost-all-dev \
      pkg-config \
      cmake \
      curl \
      git \
      tar \
      wget \
      python3-dev \
      python3-venv \
      doxygen \
      graphviz \
      libgraphviz-dev \
      libsqlite3-dev \
      libeigen3-dev \
      pybind11-dev \
      libfmt-dev \
      librange-v3-dev

RUN apt-get -y install nlohmann-json3-dev

COPY . /delphi
WORKDIR /delphi

RUN mkdir -p data && curl http://vanga.sista.arizona.edu/delphi_data/delphi.db -o data/delphi.db

RUN git clone https://github.com/meltwater/served
RUN cd served && mkdir build && cd build && cmake .. && make -j install && cd ../..
RUN mkdir build && cd build && cmake .. && make -j

CMD delphi_rest_api
