[![Documentation Status](https://readthedocs.org/projects/delphi-framework/badge/?version=latest)](http://delphi-framework.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/ml4ai/delphi.svg?branch=master)](https://travis-ci.org/ml4ai/delphi)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/ml4ai/delphi/master)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1436914.svg)](https://doi.org/10.5281/zenodo.1436914)

<img src="https://raw.githubusercontent.com/ml4ai/delphi/master/docs/delphi_logo.png" width="250">

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

Delphi is a framework for assembling, exporting and executing executable DBN
(dynamic Bayesian Network) models built for the DARPA World Modelers Program.

For a detailed description of our procedure to convert text to models, see
[this document](http://vision.cs.arizona.edu/adarsh/export/Arizona_Text_to_Model_Procedure.pdf)

For API documentation, see [delphi.readthedocs.io](https://delphi.readthedocs.io).

# Citing

If you use Delphi, please cite the following:

```latex
@misc{Delphi,
    Author = {Adarsh Pyarelal and Paul Hein and Clayton Morrison},
    Title = {Delphi: A Framework for Assembling Causal Probabilistic Models from Text and Software.},
    doi={10.5281/zenodo.1436915},
}
```

The general use case involves the following steps:

1. Creating a dynamic Bayes network model from a set of INDRA statements, which
   are data structures that represent knowledge fragments about causal relations
   that are extracted using machine reading software such as Eidos.

2. Executing the model. This produces a set of time evolution
   sequences that are sampled stochastically from the conditional probability
   distributions constructed in step 1.

See the Usage section for more details.

# Installation

## Requirements

- Python 3.6 or higher.

The recommended way to install Delphi is to use the `pip` Python package
manager:

```bash
pip install -e git+https://github.com/ml4ai/delphi.git#egg=delphi
```

## MacOS installation notes

If using Homebrew to install graphviz, then to install pygraphviz using `pip`,
do:

```bash
pip install --install-option="--include-path=/usr/local/include/" \
            --install-option="--library-path=/usr/local/lib" pygraphviz
```

Then pip install pipenv and do:

```bash
pipenv install -d
```

## Ubuntu installation notes
To install graphviz on Ubuntu, do

```bash
sudo apt-get install graphviz libgraphviz-dev pkg-config
```
## Configuration file specification
In order to access ancillary data used by delphi please use the following command to create a `config.json` file from the included `master_config.json`
```bash
cp master_config.json delphi/config.json
```
Once you have created the `config.json` file please update the path components to allow delphi to find your local copy of any ancillary data you seek to use with delphi. The specification of which dataset each path should point to are outlined below:
- `delphi_data_path`: This should point to the data directory hosted in the the [Delphi Google Drive folder](https://drive.google.com/drive/u/1/folders/1XznXUzqVIDQKuvgZuTANRy10Q2I1CqQ6)
- `dssat_repo_path`: This should point to your local checkout of the [DSSAT](https://github.com/DSSAT/dssat-csm) repository
- `ed2_repo_path`: This should point to your local checkout of the [Ecosystem Demography Model](https://github.com/EDmodel/ED2) repository
- `clm_repo_path`: This should point to your local checkout of the clm portion of the [Community Earth System Model](https://github.com/ESCOMP/cesm#id5) repository. Note that the path component to go from the root of `cesm/` to `clm/` has already been provided in `master_config.json`


# Usage

## Jupyter notebook workflow

Please see `notebooks/PI Meeting 2018 Demo.ipynb` for an example analysis
workflow using a Jupyter notebook.

You can also use the [Delphi binder](https://mybinder.org/v2/gh/ml4ai/delphi/master)
to try out the Jupyter notebook demo without having to install Delphi locally.

You can see a prerendered HTML version of the notebook at
[http://vision.cs.arizona.edu/adarsh/Delphi-Demo-Notebook.html](http://vision.cs.arizona.edu/adarsh/Delphi-Demo-Notebook.html)


## Command line usage


In the following sections, we will go into more detail on model creation and
execution.

### Create model

To create a model from a set of INDRA statements, do

```bash
delphi create
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


```csv
rainfall,100.0
∂(rainfall)/∂t,1.0
crop yield,100.0
∂(crop yield)/∂t,1.0
```

### Execute model

To execute the model, do:

```bash
delphi execute
```

This takes as input the files `dressed_CAG.pkl` and `variables.csv` and creates
an output file `output_sequences.csv` (these are the default input and output
filenames, but they can be changed with command line flags). that looks like
this:


```csv
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
