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

