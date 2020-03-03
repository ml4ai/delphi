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
      python3.7 \
      python3.7-venv \
      doxygen \
      openjdk-8-jdk \
      libgraphviz-dev \
      graphviz \
      nlohmann-json3-dev \
      libsqlite3-dev \
      libboost-all-dev \
      libeigen3-dev \
      libspdlog-dev \
      pybind11-dev \
      libfmt-dev \
      librange-v3-dev \
    && git clone https://github.com/ml4ai/delphi \
    && curl http://vanga.sista.arizona.edu/delphi_data/delphi.db -o delphi/data/delphi.db \
    && curl http://vanga.sista.arizona.edu/delphi_data/model_files.tar.gz -o delphi/data/model_files.tar.gz \
    && git config --global user.email "adarsh@email.arizona.edu" \
    && git config --global user.name "Adarsh Pyarelal"


# Set up virtual environment
ENV VIRTUAL_ENV=/delphi_venv
RUN python3.7 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set the environment variable DELPHI_DB to point to the SQLite3 database.
ENV DELPHI_DB=/delphi/data/delphi.db
ENV MODEL_FILES=/delphi/data/source_model_files
WORKDIR /delphi
RUN pip install wheel && \
    pip install scipy \
      matplotlib \
      pandas \
      seaborn \
      sphinx \
      sphinx-rtd-theme \
      recommonmark \
      ruamel.yaml \
      breathe \
      exhale \
      pytest \
      pytest-cov \
      pytest-sugar \
      pytest-xdist \
      plotly \
      sympy \
      flask \
      flask-WTF \
      flask-codemirror \
      salib \
      torch \
      tqdm \
      SQLAlchemy \
      flask-sqlalchemy \
      flask-executor \
      python-dateutil \
  && make extensions
