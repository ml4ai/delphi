#!/usr/bin/env bash
# docker pull pauldhein/project-containers:latest
docker build --cache-from pauldhein/project-containers:latest --tag pauldhein/project-containers:latest .
docker run -itd --rm --name build-con pauldhein/project-containers:latest
docker exec build-con git fetch
docker exec build-con git checkout 48c5884451cc74155dde3cc6f2a237347400f956
# docker exec build-con pip install -e .[test,docs]
docker exec build-con make test
docker exec -w /repo/delphi/docs build-con make apidocs
docker exec -w /repo/delphi/docs build-con make html
docker cp build-con:/repo/delphi/docs ~/Desktop
