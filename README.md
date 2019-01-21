[![Documentation Status](https://readthedocs.org/projects/delphi/badge/?version=latest)](http://delphi.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/ml4ai/delphi.svg?branch=master)](https://travis-ci.org/ml4ai/delphi)
[![Coverage Status](https://coveralls.io/repos/github/ml4ai/delphi/badge.svg?branch=master)](https://coveralls.io/github/ml4ai/delphi?branch=master)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/ml4ai/delphi/master)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1436914.svg)](https://doi.org/10.5281/zenodo.1436914)

<img src="https://raw.githubusercontent.com/ml4ai/delphi/master/docs/delphi_logo.png" width="250">

# Delphi

## Contents
- [Citing](#citing)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

Modeling complex phenomena such as food insecurity requires reasoning
over multiple levels of abstraction and fully utilizing expert
knowledge about multiple disparate domains, ranging from the
environmental to the sociopolitical.

Delphi is a framework for assembling causal, dynamic, probabilistic
models from information extracted from two sources:

- *Text*: Delphi utilizes causal relations extracted using machine
   reading from text sources such as UN agency reports, news articles,
   and technical papers.

- *Software*: Delphi also incorporates functionality to extract
   abstracted representations of scientific models from code that
   implements them, and convert these into probabilistic models.

For a detailed description of our procedure to convert text to models,
see [this
document](http://vision.cs.arizona.edu/adarsh/export/Arizona_Text_to_Model_Procedure.pdf).

For API documentation, see
[delphi.readthedocs.io](https://delphi.readthedocs.io).

Delphi is also part of the
[AutoMATES](https://ml4ai.github.io/automates/) project.

## Citing

If you use Delphi, please cite the following:

```
@misc{Delphi,
    Author = {Adarsh Pyarelal and Paul Hein and Clayton Morrison},
    Title = {Delphi: A Framework for Assembling Causal Probabilistic Models from Text and Software.},
    doi={10.5281/zenodo.1436915},
}
```


## Installation

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
  If you have another version of Python already installed and need it for other
  projects, we recommend [`pyenv`](https://github.com/pyenv/pyenv) to manage
  multiple versions of Python.

- [Graphviz](https://www.graphviz.org/download/) - Delphi uses this to
  visualize causal analysis graphs.

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

- Installing [INDRA](https://github.com/sorgerlab/indra`): We
  currently recommend installing the latest INDRA rather than through
  PyPI. To install the latest, execute the following from the
  terminal:
    ```bash 
    pip install git+https://github.com/sorgerlab/indra.git
    ```

### MacOS Installation Notes

#### Installing Graphviz on MacOS

This can be done using [Homebrew](https://brew.sh): 

```bash
brew install graphviz
``` 

If you use Homebrew to install graphviz, then you when you install
pygraphviz by pip, you need to install it by specifying paths to grab
the necessary brew-based include and libs, as done below:

```bash
pip install --install-option="--include-path=/usr/local/include/" \
            --install-option="--library-path=/usr/local/lib" pygraphviz
```

#### Installing Tanget on MacOS X with python >=3.7

The model analysis (AutoMATES-related) portion of delphi now depends
on [Tangent](https://github.com/google/tangent), which in turn depends
on a library in TensorFlow, which itself does not support (yet) python
`>=3.7`. You can still fix the install to tangent (after following the
`pip install .` instructions above) by doing the following (if you use
a virtual environment for delphi work (recommended!), then be sure to
do the follwoing while within the virutal environment):

- Clone the tanget project. `cd` to location where you would like the
    tangent source to be cloned and then do: 
    ```
    git clone https://github.com/google/tangent.git
    ```
- `cd` into tangent
- Run the tangent installer:
    ```
    python setup.py install
    ```

### Ubuntu installation notes
To install graphviz on Ubuntu, do

```bash
sudo apt-get install graphviz libgraphviz-dev pkg-config
```

### Environment variables

To parameterize Delphi models correctly, you will need to set the `DELPHI_DB`
environment variable to the path to your local copy of the Delphi SQLite
database, which you can download with:

```
curl http://vision.cs.arizona.edu/adarsh/delphi.db -o delphi.db
```

*Optional*:

If you are working on program analysis, you may want to optionally set the
following environment variables as well.
- `DSSAT_REPOSITORY`: This should point to your local
  checkout of the [DSSAT](https://github.com/DSSAT/dssat-csm) repository.
- `ED2_REPOSITORY`: This should point to your local checkout of the [Ecosystem
  Demography Model](https://github.com/EDmodel/ED2) repository.

### Building documentation

To build and view the documentation, run the following commands from the root of
the repository:

```
make docs; open docs/_build/html/index.html
```

(On a Linux system, replace `open` with `xdg-open`)

## Usage

### Jupyter notebook workflow

Please see `notebooks/Delphi-Demo-Notebook.ipynb` for an example analysis
workflow using a Jupyter notebook.

You can also use the [Delphi binder](https://mybinder.org/v2/gh/ml4ai/delphi/master)
to try out the Jupyter notebook demo without having to install Delphi locally.

You can see a prerendered HTML version of the notebook 
[here.](http://vision.cs.arizona.edu/adarsh/Delphi-Demo-Notebook.html)


### Command line usage

In the following sections, we will go into more detail on model execution.

### Execute model

To execute the model, do:

```bash
delphi execute
```

This takes as input the files `dressed_CAG.pkl` which contains an AnalysisGraph object,
and `bmi_config.txt`, which looks like this:

```
rainfall,100.0
∂(rainfall)/∂t,1.0
crop yield,100.0
∂(crop yield)/∂t,1.0
```

and creates an output file `output_sequences.csv` (these are the default input
and output filenames, but they can be changed with command line flags), that
looks like this:

```
seq_no,time_slice,rainfall,crop yield
0,0,100.0,100.0
0,1,102.60446042864127,102.27252764173306
0,2,103.68597583717079,103.90533882812889
1,0,100.0,100.0
1,1,102.16123221277232,101.92000855752877
1,2,103.60428897964772,101.7157053024733
```

- `seq_no` specifies the sampled sequence
- `time_slice` denotes the time slice of the sequence
- The labels of the other columns denote the factors in the CAG. By collecting
    values from the same time slice over multiple sequences, one can create a
    histogram for the value of a quantity of interest at a particular time
    point. The spread of this histogram represents the uncertainty in our
    estimate.

To see all the command line options and the help message, do `delphi -h`.

## License

Delphi is licensed under the Apache License 2.0.
