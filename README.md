[![Documentation Status](https://readthedocs.org/projects/delphi-framework/badge/?version=latest)](http://delphi-framework.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/ml4ai/delphi.svg?branch=master)](https://travis-ci.org/ml4ai/delphi)
[![Coverage Status](https://coveralls.io/repos/github/ml4ai/delphi/badge.svg?branch=master)](https://coveralls.io/github/ml4ai/delphi?branch=master)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/ml4ai/delphi/master)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1436914.svg)](https://doi.org/10.5281/zenodo.1436914)

<img src="https://raw.githubusercontent.com/ml4ai/delphi/master/docs/delphi_logo.png" width="250">

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

Delphi is a framework for assembling, exporting and executing executable DBN
(dynamic Bayesian Network) models built for the DARPA World Modelers Program.

The general use case involves the following steps:

1. Creating a dynamic Bayes network model from a set of INDRA statements, which
   are data structures that represent knowledge fragments about causal relations
   that are extracted using machine reading software such as Eidos.

2. Executing the model. This produces a set of time evolution
   sequences that are sampled stochastically from the conditional probability
   distributions constructed in step 1.

See the Usage section for more details.

For a detailed description of our procedure to convert text to models, see
[this document](http://vision.cs.arizona.edu/adarsh/export/Arizona_Text_to_Model_Procedure.pdf)

For API documentation, see [delphi.readthedocs.io](https://delphi.readthedocs.io).

# Citing

If you use Delphi, please cite the following:

```
@misc{Delphi,
    Author = {Adarsh Pyarelal and Paul Hein and Clayton Morrison},
    Title = {Delphi: A Framework for Assembling Causal Probabilistic Models from Text and Software.},
    doi={10.5281/zenodo.1436915},
}
```


# Installation

Delphi is under active development in an academic, rather than a commercial
setting, so we do not have the resources to test it out on the Windows operating
system, or provide a one-step/one-click setup process.

That being said, this is a Python package, and we use platform-independent path
handling internally within the code using `pathlib`, so *technically*, it should
work fine on Windows machines as well, and we will try to guide you through the
installation process as much as possible. Pull requests for improvements are
always welcome.

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

- Fire up a terminal, navigate to the directory that you would like to install Delphi in, then execute the following in the terminal:
    ```bash
    git clone https://github.com/ml4ai/delphi
    cd delphi
    pip install pipenv
    pipenv install -d --skip-lock
    ```

## Installing Graphviz on MacOS

This can be done using [Homebrew](https://brew.sh):
```
brew install graphviz
```
If you use Homebrew to install graphviz, then you may need to install
pygraphviz by specifying certain paths, as done below.

```bash
pipenv install --install-option="--include-path=/usr/local/include/" \
               --install-option="--library-path=/usr/local/lib" pygraphviz
```

## Ubuntu installation notes
To install graphviz on Ubuntu, do

```bash
sudo apt-get install graphviz libgraphviz-dev pkg-config
```

## Environment variables

To parameterize Delphi models correctly, you will need to set the `DELPHI_DATA`
environment variable to the path to your local copy of the Delphi data
directory. You can download the data directory from the 
[Delphi Google Drive folder](https://drive.google.com/drive/u/1/folders/1XznXUzqVIDQKuvgZuTANRy10Q2I1CqQ6)

*Optional*:

If you are working on program analysis, you may want to optionally set the
following environment variables as well.
- `DSSAT_REPOSITORY`: This should point to your local
  checkout of the [DSSAT](https://github.com/DSSAT/dssat-csm) repository.
- `ED2_REPOSITORY`: This should point to your local checkout of the [Ecosystem
  Demography Model](https://github.com/EDmodel/ED2) repository.

# Usage

## Jupyter notebook workflow

Please see `notebooks/Delphi-Demo-Notebook.ipynb` for an example analysis
workflow using a Jupyter notebook.

You can also use the [Delphi binder](https://mybinder.org/v2/gh/ml4ai/delphi/master)
to try out the Jupyter notebook demo without having to install Delphi locally.

You can see a prerendered HTML version of the notebook 
[here.](http://vision.cs.arizona.edu/adarsh/Delphi-Demo-Notebook.html)


## Command line usage

In the following sections, we will go into more detail on model creation and
execution.

### Create model

To create a model from a set of INDRA statements, do

```bash
python delphi/cli.py create
```

Optionally combine this with the `--indra_statements` input parameter to specify
the path to the INDRA statements. By default, `--create_model` will load a
provided set of INDRA statements included as a pickled python file in
`data/curated_statements.pkl`, and generate the output files


```
./
├── delphi_cag.json
├── dressed_CAG.pkl
└── variables.csv
```

The files are
- `delphi_cag.json`: This contains the link structure of the causal analysis
- graph, along with conditional probability tables, indicators for the latent
    variables represented by the CAG nodes, and initial values for the
    indicators when available.
- `dressed_CAG.pkl`: This is a Python pickle object that can be used in Python
    programs. It contains a networkx Digraph object, with conditional
    probability distributions attached to the edges.
- `variables.csv`: This CSV file contains the names and initial values of the
    components of the latent state of the DBN, corresponding to the factors in
    the CAG and their partial derivatives with respect to time. This is set with
    some default values which can be edited prior to the execution of the model.


```
rainfall,100.0
∂(rainfall)/∂t,1.0
crop yield,100.0
∂(crop yield)/∂t,1.0
```

### Execute model

To execute the model, do:

```bash
python delphi/cli.py execute
```

This takes as input the files `dressed_CAG.pkl` and `variables.csv` and creates
an output file `output_sequences.csv` (these are the default input and output
filenames, but they can be changed with command line flags). that looks like
this:


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

# License

Delphi is licensed under the Apache License 2.0.
