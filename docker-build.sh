#!/usr/bin/env bash
docker pull pauldhein/project-containers:delphi
docker run -itd --rm --name build-con pauldhein/project-containers:delphi
docker exec build-con bash -c 'git fetch; git checkout 5f8eec2bfbe66b98e91b98ebda5db4de7e45e580'
docker exec build-con bash -c 'rm -rf /repo/delphi/build'
docker exec build-con pip install -e .[test,docs]
docker exec build-con make test
# docker exec -w /repo/delphi/docs build-con bash -c "make apidocs; make html"
