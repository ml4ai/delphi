#!/usr/bin/env bash
docker pull pauldhein/project-containers:delphi
docker run -itd --rm --name build-con pauldhein/project-containers:delphi
docker exec build-con bash -c 'git fetch; git checkout 04fedc0cb701f90112648440d704858a8d3acc72'
docker exec build-con bash -c 'rm -rf /repo/delphi/build'
docker exec build-con pip install -e .[test,docs]
docker exec build-con make test
# docker exec -w /repo/delphi/docs build-con bash -c "make apidocs; make html"
