# Installation

## Users

Delphi is under active development in an academic, rather than a
commercial setting, so we do not have the resources to test it out on
the Windows operating system, or provide a one-step/one-click setup
process.

That being said, this is a Python package, and we use
platform-independent path handling internally within the code using
`pathlib`, so *technically*, it should work fine on Windows machines
as well, and we will try to guide you through the installation process
as much as possible. Pull requests for improvements are always
welcome.

The following are the requirements for Delphi:

- Python 3.6 or higher.

  - Python 3.6 is recommended.

- [Graphviz](https://www.graphviz.org/download/) - Delphi uses this to
  visualize causal analysis graphs. See MacOS and Ubuntu notes below
  for installing graphviz.

The following installation instructions are directed at developers working on
Linux and MacOS operating systems. We assume familiarity with the following:

- The command line/terminal
- Unix commands such as `cd`.
- Environment variables.

Here are the steps for installation.

- Fire up a terminal, navigate to the directory that you would like to
  install Delphi in, then execute the following in the terminal:

    ```bash 
    git clone https://github.com/ml4ai/delphi
    cd delphi 
    pip install .
    ``` 


## Developers

If you are developing Delphi and want to run tests or compile the
documentation, then also do the following (from the root of Delphi): 

```
pip install -e .[test,docs]
```

### Graphviz installation notes

#### MacOS

This can be done using [Homebrew](https://brew.sh): 

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

### Debian

To install graphviz on Debian systems (like Ubuntu), do

```bash
sudo apt-get install graphviz libgraphviz-dev pkg-config
```

### Environment variables

There are several environment variables that need to be set in order
for Delphi to function.

#### Adding the Delphi root to the PYTHONPATH

Set the PYTHONPATH to include the absolute path to the root of the
delphi project. This can be set in one of two places:
- In your `~/.bash_profile` (Mac) or `~/.bashrc` (linux) file. For example:
    ```bash
    export PYTHONPATH="/Users/claytonm/Documents/repository/delphi:$PYTHONPATH"
    ```
- If you use a virtual environment, instead of adding yet another
    path to your global PYTHONPATH in your `~/.bash_profile` or `~/.bashrc`,
    instead you can add the path to the Delphi root to be used only in your
    virtual environment `project.pth` file. 
    This has the advantage of not polluting your global PYTHONPATH when you
    run python in other contexts (e.g., other virtual environments).
    This file is located within the virtual 
    environment as follows (the following assumes your virtual environment 
    is named `<venv>`, and `<version>` is the version number of your python, 
    such as 3.6 or 3.7):
    ```
    <venv>/lib/python<version>/site-packages/project.pth
    ```
    NOTE: you may need to create the `project.pth` if one does not already
    exist.
    To this file you simply add the absolute path to the Delphi root (you
    do not use export or `PYTHONPATH`), for example:
    ```
    /Users/claytonm/Documents/repository/delphi
    ```

#### Other environment variables

- To parameterize Delphi models correctly, you will need to set the
    `DELPHI_DB` environment variable to the path to your local copy of the
    Delphi SQLite database, which you can download with:

    ```bash
    curl http://vision.cs.arizona.edu/adarsh/delphi.db -o delphi.db
    ```

    Then set the variable enviornment (again, may be done within your bash
    resource file or the virtual envrionment `project.pth`). 
    The delphi.db name must appear at the end of the path, for example:

    ```bash
    export DELPHI_DB="/Users/claytonm/Documents/repository/delphi_db/delphi.db"
    ```


- *Optional*: If you are working on program analysis, you may want to
optionally set the following environment variables as well (again, in 
.bash_profile/.bashrc or viritual environment projects.pth).

  - `DSSAT_REPOSITORY`: This should point to your local checkout of
  the [DSSAT](https://github.com/DSSAT/dssat-csm) repository.

  - `ED2_REPOSITORY`: This should point to your local checkout of the
  [Ecosystem Demography Model](https://github.com/EDmodel/ED2)
  repository.

### Building documentation

(This requies you have performed the installation for developers, above.) 
To build and view the documentation, run the following commands from the root of
the repository:

```
make docs; open docs/_build/html/index.html
```

(On a Linux system, replace `open` with `xdg-open`)

