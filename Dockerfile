FROM        ubuntu:16.04
MAINTAINER  Paul D. Hein <pauldhein@email.arizona.edu>
CMD         bash

RUN apt-get update && apt-get install -y software-properties-common
# Required system packages
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update
RUN apt-get -y install apt-utils build-essential g++-9 curl git tar wget time


RUN apt-get -y install python3.6 python3.6-dev python3-pip python3.6-venv
RUN apt-get -y install doxygen openjdk-8-jdk graphviz libgraphviz-dev pkg-config
RUN apt-get -y install sqlite3 libsqlite3-dev libboost-all-dev libeigen3-dev

# Setup the correct version of Python and install/update pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2
RUN curl https://bootstrap.pypa.io/ez_setup.py -o - | python3.6 && python3.6 -m easy_install pip
RUN pip install pip --upgrade && pip install wheel
RUN echo  'alias python=python3' >> ~/.bashrc

WORKDIR /data
# Download SQLite3 database containing model parameterization data.
RUN curl -O http://vanga.sista.arizona.edu/delphi_data/delphi.db

# Set the environment variable DELPHI_DB to point to the SQLite3 database.
ENV DELPHI_DB=/data/delphi.db

# Build the delphi testing environment
ENV CC=gcc-9
ENV CXX=g++-9
WORKDIR /repo
RUN git clone https://github.com/ml4ai/delphi.git

WORKDIR /repo/delphi
RUN pip install cython
RUN pip install cmake
RUN pip install futures
RUN pip install -e .[test,docs]
RUN rm -rf build/
RUN make extensions
