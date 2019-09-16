#!/usr/bin/env bash
docker pull pauldhein/project-containers
docker run pauldhein/project-containers /bin/sh -c "git fetch; git checkout travis-docker-ci; git pull"
docker run pauldhein/project-containers /bin/sh -c "pip install -e .[test,docs]"

docker run pauldhein/project-containers /bin/sh -c "make test"
# docker run -w="/repo/delphi/docs" pauldhein/project-containers /bin/sh -c "make apidocs; make html"
