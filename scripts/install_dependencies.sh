#!/bin/bash

echo "Installing Delphi C++ library dependencies."

echo "Checking OS."
# If MacOS, then try MacPorts and Homebrew

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "MacOS detected. Checking for XCode developer kit."

    XCode_sdk_dir="/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs"
    if [[ ! -d "${XCode_sdk_dir}" ]]; then
        echo "No XCode developer kit found. You need to get this from the app store."
        exit 1
    else
        if [[ ! -e "${XCode_sdk_dir}/MacOSX10.14.sdk" ]]; then
            if [[ ! -e "${XCode_sdk_dir}/MacOSX.sdk" ]]; then
                echo "No MacOSX.sdk in ${XCode_sdk_dir}."
                echo "Possibly MacOS has changed things (again)."
                exit 1
            else
                pushd "${XCode_sdk_dir}" > /dev/null
                    echo "Linking missing MacOSX10.14.sdk to MacOSX.sdk in ${XCode_sdk_dir}/"
                    sudo ln -s "MacOSX.sdk" "MacOSX10.14.sdk"
                    if [[ $? -ne 0 ]]; then exit 1; fi;
                popd > /dev/null
            fi
        fi
    fi

    echo "Found XCode developer kit."
    echo "Checking for MacPorts or Homebrew package managers."

    if [ -x "$(command -v port)" ]; then
        echo "'port' executable detected, assuming that MacPorts"\
        "(https://www.macports.org) is installed and is the package manager."

        echo "Installing Delphi dependencies using MacPorts."

        sudo port selfupdate
        if [[ $? -ne 0 ]]; then exit 1; fi;

        sudo port -N install \
            cmake \
            libfmt \
            doxygen \
            nlohmann-json \
            eigen3 \
            pybind11 \
            boost\
            range-v3
        if [[ $? -ne 0 ]]; then exit 1; fi;

    elif [ -x "$(command -v brew)" ]; then
        echo "\'brew\' executable detected, assuming that Homebrew"\
        "\(https://brew.sh\) is installed and is the package manager."

        echo "Installing Delphi dependencies using Homebrew."

        brew update
        if [[ $? -ne 0 ]]; then exit 1; fi;

        brew install \
          cmake \
          fmt \
          doxygen \
          boost \
          range-v3 \
          nlohmann-json \
          pybind11 \
          eigen

        if [[ ! -z $TRAVIS ]]; then
          # On Travis, we will install lcov to provide code coverage estimates.
          brew install lcov;
        fi;
    else
        echo "No package manager found for $OSTYPE"
        exit 1
    fi
else
    echo "Neither Homebrew nor MacPorts found. We cannot proceed."
    exit 1
fi

echo "Delphi dependency installation complete."
echo " "
