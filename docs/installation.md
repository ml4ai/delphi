# Installation

## Prerequisites

The following are the prerequisites for Delphi:

- Python 3.6+ and pip
- [Graphviz](https://www.graphviz.org/download/) - Delphi uses this to
  visualize causal analysis graphs. See MacOS and Ubuntu notes below
  for installing graphviz.
- A C++17-compatible compiler.
- [Boost](https://www.boost.org)
- [CMake](https://cmake.org)

## Step-by-step instructions

1. Download and set up the Delphi database
   ```
   curl -O http://vanga.sista.arizona.edu/delphi_data/delphi.db
   ```
   
   

   Then, point the environment variable `DELPHI_DB` to point to `delphi.db`.
   
   If on Linux, add this to your ~/.bashrc:
 
   ```
   create_new_venv() {
     mkdir -p ~/.venvs
     python -m venv ~/.venvs/$1
   }

   # Usage example:
   #
   #     activate_py3 37
   #
   # will activate Python 3.7
   activate_py3() {
     sudo port select --set python python$1
     sudo port select --set python3 python$1
     sudo port select --set pip pip$1
     sudo port select --set pip3 pip$1
   }

   # Activate a Python virtual environment
   activate() {
     source ~/.venvs/$1/bin/activate
   }
   
   export DELPHI_DB=/Users/<user>/delphi.db
   ```
   
   And then source the file
   
   ```
   source ~/.bashrc
   ```

   If on a Mac, follow the same steps as above, using file 
   
   ```
   ~/.bash_profile
   ```

2. Install Delphi using pip:
  - If you are an _end-user_:
    ```
    pip install https://github.com/ml4ai/delphi/archive/master.zip
    ```
  - If you are a Delphi _developer_, create a fresh Python virtual environment,
    activate it, and then run the following commands:
    ```
    git clone https://github.com/ml4ai/delphi
    cd delphi
    pip install -e .[test,docs]
    ```
  - To test if everything is set up properly, run the tests:
    ```
    make test_wm
    ```


#### MacOS

If you use the [Homebrew](https://brew.sh) package manager:

```bash
brew install graphviz
``` 

If you use Homebrew to install graphviz, you may need to install pygraphviz by pip, specifying paths to grab
the necessary brew-based include and libs, as done below:

```bash
pip install --install-option="--include-path=/usr/local/include/" \
            --install-option="--library-path=/usr/local/lib" pygraphviz
```

If you use [MacPorts](https://www.macports.org/install.php) package manager:

```bash
port install graphviz
``` 
Installation using pip

```bash
pip install --install-option="--include-path=/opt/local/include/" \
            --install-option="--library-path=/opt/local/lib" pygraphviz
```



### Debian

To install graphviz on Debian systems (like Ubuntu), do

```bash
sudo apt-get install graphviz libgraphviz-dev pkg-config
```

### Completing the installation

Start the delphi python virtual environment:

```bash
activate delphi
```

With the delphi python virtual environment running, install the test and documentation files:

```bash
sudo port install pybind11
pip install -e .[test,docs]
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
