#!/usr/bin/env bash

docker pull pauldhein/project-containers
docker run -d -p 127.0.0.1:80:4567 pauldhein/project-containers /bin/sh -c "git clone https://github.com/ml4ai/delphi.git /"
docker ps -a
docker run pauldhein/project-containers /bin/sh -c "cd /repos/delphi; git checkout -qf travis-docker-ci"
docker run pauldhein/project-containers /bin/sh -c "make test"
docker run pauldhein/project-containers /bin/sh -c "cd /delphi/delphi/docs; make apidocs; make html"
