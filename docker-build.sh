#!/usr/bin/env bash
docker pull pauldhein/project-containers:delphi
docker run -itd --rm --name build-con pauldhein/project-containers:delphi
docker exec build-con git fetch
docker exec build-con git checkout 04fedc0cb701f90112648440d704858a8d3acc72
docker exec build-con rm -rf /repo/delphi/build
docker exec build-con pip install -e .[test,docs]
docker exec build-con make test
docker cp build_docs.sh build-con:/repo
docker exec  build-con /repo/build_docs.sh
# docker exec -w /repo/delphi/docs build-con make apidocs
# docker exec -w /repo/delphi/docs build-con make html
docker cp build-con:/repo/delphi/docs ~/Desktop
