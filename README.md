[![Documentation Status](https://readthedocs.org/projects/delphi-framework/badge/?version=latest)](http://delphi-framework.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/ml4ai/delphi.svg?branch=master)](https://travis-ci.org/ml4ai/delphi)

# Delphi

- [Installation](#installation)
- [Usage](#usage)
- [Visualization (deprecated since v2.0.0)](#cag-visualization)
- [License](#license)

Delphi is a framework for assembling, exporting and executing executable DBN
(dynamic Bayesian Network) models built for the DARPA World Modelers Program.

The followng instructions describe how to install and use Delphi.

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

# Usage

To see all the options and the help message, do `./delphi.py`, which produces
the following output:

```
usage: delphi.py [-h] [--create_model] [--create_cra_cag]
                 [--indra_statements INDRA_STATEMENTS]
                 [--adjective_data ADJECTIVE_DATA] [--execute_model] [--dt DT]
                 [--n_statements N_STATEMENTS] [--steps STEPS]
                 [--samples SAMPLES] [--output_sequences OUTPUT_SEQUENCES]
                 [--output_cag_json OUTPUT_CAG_JSON]
                 [--input_variables_path INPUT_VARIABLES_PATH]
                 [--output_variables_path OUTPUT_VARIABLES_PATH]
                 [--input_dressed_cag INPUT_DRESSED_CAG]
                 [--output_dressed_cag OUTPUT_DRESSED_CAG]

Dynamic Bayes Net Executable Model

optional arguments:
  -h, --help            show this help message and exit
  --create_model        Export the dressed CAG as a pickled object, the
                        variables with default initial values as a CSV file,
                        and the link structure of the CAG as a JSON file.
                        (default: False)
  --create_cra_cag      Export CAG in JSON format for Charles River Analytics
                        (default: False)
  --indra_statements INDRA_STATEMENTS
                        Pickle file containing INDRA statements (default:
                        data/sample_indra_statements.pkl)
  --adjective_data ADJECTIVE_DATA
                        Path to the gradable adjective data file. (default:
                        data/adjectiveData.tsv)
  --execute_model       Execute DBN and sample time evolution sequences
                        (default: False)
  --dt DT               Time step size (default: 1.0)
  --n_statements N_STATEMENTS
                        Number of INDRA statements to take from the pickled
                        object containing them (default: 5)
  --steps STEPS         Number of time steps to take (default: 5)
  --samples SAMPLES     Number of sequences to sample (default: 100)
  --output_sequences OUTPUT_SEQUENCES
                        Output file containing sampled sequences (default:
                        dbn_sampled_sequences.csv)
  --output_cag_json OUTPUT_CAG_JSON
                        Path to the output CAG JSON file (default:
                        delphi_cag.json)
  --input_variables_path INPUT_VARIABLES_PATH
                        Path to the variables of the input dressed cag
                        (default: variables.csv)
  --output_variables_path OUTPUT_VARIABLES_PATH
                        Path to the variables of the output dressed cag
                        (default: variables.csv)
  --input_dressed_cag INPUT_DRESSED_CAG
                        Path to the input dressed cag (default:
                        dressed_CAG.pkl)
  --output_dressed_cag OUTPUT_DRESSED_CAG
                        Path to the output dressed cag (default:
                        dressed_CAG.pkl)

```

In the following sections, we will go into more detail on model creation and
execution.

## Create model

To create a model from a set of INDRA statements, use the invocation:

```bash
./delphi.py --create_model
```

Optionally combine this with the `--indra_statements` input parameter to specify
the path to the INDRA statements. By default, `--create_model` will load a
provided set of INDRA statements included as a pickled python file in
`data/sample_indra_statements.pkl`, and generate the output files 


```
./
├── cag.json
├── dressed_CAG.pkl
└── variables.csv
```

The files are
- `cag.json`: This contains the link structure of the causal analysis graph.
    Here is an example JSON file representing a CAG with the link `rainfall →
    crop yield`. 

```json
 {
  "name": "Dynamic Bayes Net Model",
  "dateCreated": "2018-04-25 15:27:32.230457",
  "variables": [
    {
      "name": "rainfall",
      "units": "units",
      "dtype": "real",
      "arguments": []
    },
    {
      "name": "crop yield",
      "units": "units",
      "dtype": "real",
      "arguments": [
        "rainfall"
      ]
    }
  ]
} 
```

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

## Execute model

To execute the model, do

```bash
./delphi.py --execute_model
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


# CAG visualization

Since version 2.0.0, Delphi no longer provides functionality for visualizing
CAGs, instead, this functionality has been integrated into the INDRA framework.
See below for a complete example.

![alt text](/docs/delphi_example.png?raw=True")

# License 

Delphi is licensed under the Apache License 2.0.
