FROM        ubuntu:18.04
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
      python3 \
      doxygen \
      openjdk-8-jdk \
      libgraphviz-dev \
      graphviz \
      libsqlite3-dev \
      libboost-all-dev \
      libeigen3-dev \
      libfmt-dev \
      sudo \
    && git clone https://github.com/ml4ai/delphi \
    && curl http://vanga.sista.arizona.edu/delphi_data/delphi.db -o delphi/data/delphi.db \
    && curl http://vanga.sista.arizona.edu/delphi_data/model_files.tar.gz -o delphi/data/model_files.tar.gz \
    && git config --global user.email "adarsh@email.arizona.edu" \
    && git config --global user.name "Adarsh Pyarelal"


# Set up virtual environment
ENV VIRTUAL_ENV=/delphi_venv
RUN apt-get install -y python3-venv
RUN python3.6 -m venv $VIRTUAL_ENV
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
      python-dateutil\
&& ./scripts/install_cmake_from_source\
&& ./scripts/install_nlohmann_json_from_source\
&& ./scripts/install_range-v3_from_source\
&& ./scripts/install_pybind11_from_source
