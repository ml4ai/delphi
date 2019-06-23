# Installation

## Prerequisites

The following are the prerequisites for Delphi:

- Python 3.6+ and pip
- [Graphviz](https://www.graphviz.org/download/) - Delphi uses this to
  visualize causal analysis graphs. See MacOS and Ubuntu notes below
  for installing graphviz.
- A C++17-compatible compiler. Delphi has been tested with Clang 8 and G++ 8.
- [Boost](https://www.boost.org)
- [CMake](https://cmake.org)

## Step-by-step instructions

1. Download and set up the Delphi database
   ```
   curl -O http://vision.cs.arizona.edu/adarsh/delphi.db
   ```

   Then, point the environment variable `DELPHI_DB` to point to `delphi.db`. On
   Linux, you can do the following:

   ```
   echo "export DELPHI_DB=`pwd`/delphi.db" >> ~/.bashrc
   source ~/.bashrc
   ```

   If on a Mac, replace `~/.bashrc` with `~/.bash_profile`.

2. Install Delphi using pip:
  - If you are an _end-user_:
    ```
    pip install https://github.com/ml4ai/delphi/archive/master.zip
    ```
  - If you are a Delphi _developer_, create a fresh Python virtual environment,
    activate it, and then run the following commands:
    ```
    git clone --recursive https://github.com/ml4ai/delphi
    cd delphi
    pip install -e .[test,docs]
    ```
  - To test if everything is set up properly, run the tests:
    ```
    make test
    ```

### Graphviz installation notes

#### MacOS

If you use the [Homebrew](https://brew.sh) package manager:

```bash
brew install graphviz
``` 

If you use Homebrew to install graphviz, then you when you install
pygraphviz by pip, you may need to install it by specifying paths to grab
the necessary brew-based include and libs, as done below:

```bash
pip install --install-option="--include-path=/usr/local/include/" \
            --install-option="--library-path=/usr/local/lib" pygraphviz
```

If you use MacPorts, invoke the following command to install pygraphviz along
with graphviz:

```
port install pygraphviz
```

### Debian

To install graphviz on Debian systems (like Ubuntu), do

```bash
sudo apt-get install graphviz libgraphviz-dev pkg-config
```

### Building documentation

(This requires you have performed the installation for developers, above.) 
To build and view the documentation, run the following commands from the root of
the repository:

```
make docs
```

Then do the following (on MacOS) to open the docs webpage in a browser.

```
open docs/_build/html/index.html
```

(On a Linux system, replace `open` with `xdg-open`)
