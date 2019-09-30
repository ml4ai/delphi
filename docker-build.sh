#!/usr/bin/env bash
docker pull pauldhein/project-containers:latest
docker run -itd --rm --name build-con pauldhein/project-containers:latest
docker exec build-con git fetch
docker exec build-con git checkout 7eb2c66a09e6deea2a86453a9cde3755a9d96793
docker exec build-con rm -rf /repo/delphi/build
docker exec build-con pip install -e .[test,docs]
docker exec build-con make test
docker exec -w /repo/delphi/docs build-con make apidocs
docker exec -w /repo/delphi/docs build-con make html
docker cp build-con:/repo/delphi/docs ~/Desktop
