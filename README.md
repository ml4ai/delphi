[![Documentation Status](https://readthedocs.org/projects/delphi-framework/badge/?version=latest)](http://delphi-framework.readthedocs.io/en/latest/?badge=latest)

# Delphi

Delphi is a framework for executing causal analysis graphs (CAGs) for DARPA's
World Modelers program. Here is an example of what it looks like:

# Installation

## Requirements

- Python 3.6 or higher.

The recommended way to install Delphi is to use the `pip` Python package
manager:

```bash
pip install -e git+https://github.com/ml4ai/delphi.git#egg=delphi
```

# Usage

To see all the options and the help message, do:

```bash
./delphi.py
```

To create a model, do:

```bash
./delphi.py --create_model
```

This will create a directory called `dbn_model`, with the following contents. 


```shell
dbn_model/
├── cag.json
├── dressed_CAG.pkl
└── variables.csv
```

The files are
- `cag.json`: This contains the link structure of the causal analysis graph.

```json
 {
  "name": "Dynamic Bayes Net Model",
  "dateCreated": "2018-04-25 15:27:32.230457",
  "variables": [
    {
      "name": "fishing community",
      "units": "units",
      "dtype": "real",
      "arguments": []
    },
    {
      "name": "income",
      "units": "units",
      "dtype": "real",
      "arguments": [
        "fishing community"
      ]
    }
  ]
} 
```

To execute the model, do

```bash
./delphi.py --execute_model
```

This creates 

# Features

# License 

Delphi is licensed under the Apache License 2.0.
