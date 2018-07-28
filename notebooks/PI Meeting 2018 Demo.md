# *World Modelers PI Meeting Demo*

*July 30, 2018*

This is a Jupyter notebook created to showcase the design and capabilities of
the Delphi package.

This demo has been tested with the version of Delphi corresponding to the commit
hash below.

```{.python .input}
!git rev-parse HEAD
```

In this demo, we will 

- create a qualitative causal analysis graph corresponding to food security, 
- quantify it with probability distributions derived from crowdsourced
  quantifications of gradable adjectives, parameterize it, and 
- 'execute' it, that is, run forward inference for the random variables
in the model.

First, we briefly discuss our text-to-model approach. See the
slide below for a high-level overview.

```{.python .input}
from IPython.display import Image; Image('images/delphi_model.png', retina=True)
```

INDRA Statements from Demo #1
---------------------------------------

We begin
by loading up the INDRA statements that form a bridge between Demo #1
and Demo
#3. These are statements that satisfy the following criteria:

- The statement
contains at least one of the relevant concepts above.
- Both the subject and
object of the statement are grounded to the UN ontology
  with a score > 0.7.
-
Both the subject and object have attached polarities (increase/decrease).
- The
INDRA belief score for the statement is at least 0.8, roughly
  corresponding to
a requirement of two independent sources of evidence per
  statement.

```{.python .input}
%load_ext autoreload
%autoreload 2
from indra.tools.assemble_corpus import load_statements
statements = load_statements('../data/6_month_eval_filtered_sts.pkl')
```

The AnalysisGraph Class
-----------------------------

The AnalysisGraph class defines the central data structure and user interface
for Delphi. It starts out as a qualitative description of the system of
interest, but it gradually morphs into an executable model, which can be
parameterized appropriately.

In the next
code block, we instantiate an AnalysisGraph object using the INDRA
statements
from Demo #1.

```{.python .input}
from delphi.AnalysisGraph import AnalysisGraph
G = AnalysisGraph.from_statements(statements)
```

Let us visualize this graph, highlighting a particular concept of interest to
us: food insecurity.

```{.python .input}
from delphi.visualization import visualize
concept_of_interest='food_insecurity'
visualize(G, nodes_to_highlight=['food_insecurity'])
```

Merging nodes
-------------

We see that `food_security` and `food_insecurity`
are separate nodes in this
graph. We perform 'linguistically-licensed'
normalization, merging them, with
appropriate polarity flipping of the relevant
INDRA statements.

```{.python .input}
from delphi.manipulation import merge_nodes
merge_nodes(G, 'food_security', concept_of_interest, same_polarity=False)
visualize(G, nodes_to_highlight=[concept_of_interest], rankdir='TB')
```

Focusing the graph on food security
-----------------------------------

Note
that there are many 'orphan' nodes in the above graph. What if we just
wanted to
study the nodes that could influence food insecurity? In the code
block below,
we execute a command that visualizes the subgraph consisting of the
ancestors of
the food security node.

```{.python .input}
from delphi.subgraphs import get_subgraph_for_concept
visualize(get_subgraph_for_concept(G, concept_of_interest),
  nodes_to_highlight=[concept_of_interest], rankdir='LR')
```

Focusing on analysis thread #3
------------------------------------

This graph
seems reasonable enough. For this demo, we will further restrict
ourselves to
scenarios described in analysis thread #3 described in the MITRE
storyboard for
South Sudan.

The next code block restricts our analysis graph to the edges that
comprise all
the paths between three particular nodes of interest: *food
insecurity*,
*conflict*, and *precipitation*.

```{.python .input}
from delphi.subgraphs import get_subgraph_for_concept_pairs
concepts_of_interest = [concept_of_interest, 'conflict', 'precipitation']
G = get_subgraph_for_concept_pairs(G, concepts_of_interest)
visualize(G, nodes_to_highlight=concepts_of_interest, rankdir='TB')
```

Graph inspection
----------------

The AnalysisGraph class has a number of
useful methods to inspect and edit the
causal analysis graph (CAG).

The first
one, shown below, is the `inspect_edge` method, which creates a
convenient table
that shows the INDRA statements that were used to construct a
particular edge in
the CAG. For example, let us take a look at the statements
associated with the
edge from *intervention* to *conflict*.

```{.python .input}
import pandas as pd
from delphi.inspection import inspect_edge
pd.options.display.max_colwidth=1000
pd.options.display.width=1000
inspect_edge(G, 'intervention', 'conflict')
```

Graph manipulation
------------------

It seems like the second statement is a
dud. Since AnalysisGraph subclasses the
[NetworkX
DiGraph](https://networkx.github.io/documentation/stable/reference/classes/digraph.html)
class, it inherits all its convenient graph accession and manipulation
interfaces.  Let's go ahead and delete the offending statement.

```{.python .input}
# Delete individual statements after inspecting edge
del G['intervention']['conflict']['InfluenceStatements'][1]
```

Let's take a look at another edge, this time between *crisis* and *conflict*.

```{.python .input}
inspect_edge(G, 'crisis', 'conflict')
```

As it turns out, the sole statement that creates this is not so good after all -
let's get rid of the edge altogether.

```{.python .input}
# Remove single edge
G.remove_edge('crisis', 'conflict')
```

We can scale this up - let's take a look at all the statements in this
AnalysisGraph - this will certainly be impractical for very large graphs, but
this is just to demonstrate bulk edge deletion.

```{.python .input}
# Set up viewing options for the notebook

from delphi.jupyter_tools import create_statement_inspection_table
from delphi.inspection import statements
create_statement_inspection_table(list(statements(G)))
```

Skimming through this table, it is apparent that some edges simply must go,
particularly ones that indicate causal relations between socio-economic and
weather phenomena (not that such an influence would be impossible, just very
unlikely on the sub-annual time scales that we will concern ourselves with).

```{.python .input}
# Remove multiple edges
G.remove_edges_from([
    ('food_insecurity', 'precipitation'),
    ('inflation', 'precipitation'),
    ('inflation', 'conflict'),
])
create_statement_inspection_table(statements(G))
```

Let us further refine this collection of statements. First, let us get rid of
the statements that are incorrectly grounded, or at least not with the
specificity required for quantitative modeling.

```{.python .input}
incorrectly_grounded_stmt_indices=[0, 1, 4, 7, 9, 10, 12, 16, 18, 20]
sts = list(statements(G))
sts = [sts[i] for i in range(len(sts)) if i not in incorrectly_grounded_stmt_indices]
create_statement_inspection_table(sts)
```

Let us now fix up the statements that have the right grounding, but the wrong
causal direction.

```{.python .input}
reverse_causal_direction_indices= [8]

def reverse_causal_direction(s):
    s.subj, s.obj = s.obj, s.subj
    s.subj_delta, s.obj_delta = s.obj_delta, s.subj_delta
    
for i in reverse_causal_direction_indices:
    reverse_causal_direction(sts[i])
    
create_statement_inspection_table(sts)
```

Inspecting the surviving statements, we see that even though the groundings are
now good, some of the polarities need to be fixed. However, we need to be a bit
careful in doing so, since the `food_security` node is now merged into the
`food_insecurity` node. In addition, we will choose to interpret 'worsening'
conflict as 'increasing' conflict.

```{.python .input}
flip_subj_polarity_indices = []
flip_obj_polarity_indices = [0, 8, 10]

for i in flip_subj_polarity_indices:
    sts[i].subj_delta['polarity'] = -sts[i].subj_delta['polarity']
    
for i in flip_obj_polarity_indices:
    sts[i].obj_delta['polarity'] = -sts[i].obj_delta['polarity']
    
create_statement_inspection_table(sts)
```

Let us create a fresh AnalysisGraph object using this manually curated set of
statements.

```{.python .input}
G = AnalysisGraph.from_statements(sts)
visualize(G, rankdir='LR', nodes_to_highlight=concepts_of_interest)
```

Concepts & Indicators
------------------------------

We then map each concept
to an 'indicator'. Indicators serve as the concrete, measurable quantities that
act as 'proxies' for the latent variables. For example, while we cannot directly
measure food security, we *can* measure the average number of kilocalories
consumed per day in a region.

Observation Model
-----------------

Let $n_{ij}$
be value of the $j^{th}$ indicator for the $i^{th}$ component of a
latent state
vector $\mathbf{s}$.  Define the probability distribution for
sampling the value
of the indicator:

$$n_{ij}\sim \mathcal{N}(s_i\mu_{ij}, \sigma_{ij})$$

where
$\mu_{ij}, \sigma_{ij}$ are the mean and standard deviation of the
indicator
$n_{ij}$.

Mapping concepts to indicators
------------------------------

The
mapping of concepts to indicators is done in a semiautomated way. The
concepts
and indicators are first mapped to vectors in a high-dimensional vector
space
(also known as word embedding). This is followed by the calculation of
pairwise
similarity scores based on the cosine of the angle between the vectors.

In the
code block below, we map concepts to indicators, and visualize the
indicators as
blue nodes.

```{.python .input}
from delphi.quantification import map_concepts_to_indicators
map_concepts_to_indicators(G, 2);  visualize(G, indicators=True)
```

Note that some of the edges between the concepts in the CAG are less opaque than
others. This is by design - the opacity of each maroon edge is proportional to
the amount of evidence there is for that edge. In this case, we use the number
of Influence statements in the edge as a proxy for the amount of evidence.

In
the code block below, we manually prune the indicators so that there is only
one
indicator per concept, and no indicators are shared among concepts. (Our
model
does not yet incorporate shared indicators.) We also associate an
indicator,
`PRECIPITATION` to the concept `precipitation`, and specify the data
source as
output from the`CYCLES` crop model .

```{.python .input}
from delphi.types import Indicator
G.nodes['conflict']['indicators'] = G.nodes['conflict']['indicators'][1:]
G.nodes['inflation']['indicators'] = G.nodes['inflation']['indicators'][1:]
G.nodes['intervention']['indicators'] = G.nodes['intervention']['indicators'][1:]
G.nodes['crisis']['indicators'] = None 
G.nodes['precipitation']['indicators'] = [Indicator('PRECIPITATION', 'CYCLES')]
visualize(G, indicators=True)
```

Once we have settled on a set of indicators to represent the abstract concepts,
we set about parameterizing the analysis graph, by specifying a year to get data
for these indicators for. The dataset used to parameterize the analysis graph
has been constructed using data from the FAOSTAT and WDI databases, as well as
output data from the `CYCLES` crop model. If no data is found for an indicator,
it is discarded from the analysis graph.

```{.python .input}
from datetime import datetime
from delphi.parameterization import parameterize
date = datetime(2014, 1, 1)
parameterize(G, date)
```

We visualize the parameterized analysis graph, with the values of the indicators
for the specified year (in this case, 2014).

```{.python .input}
visualize(G, indicators=True, indicator_values = True,
            graph_label=f'Causal Analysis Graph for South Sudan, {date.year}')
```

'Dressing' the CAG with probability distributions
-------------------------------------------------

We construct probability
distributions from the crowdsourced gradable adjective
quantification data,
thereby converting our analysis graph into a probabilistic
model.

```{.python .input}
G.infer_transition_model(res =3000)
```

Integration with CRA and SRI
----------------------------

This model can be
then exported. Since it is a Python object, it can be exported
as a
[pickled](https://docs.python.org/3/library/pickle.html) object.
Alternatively,
it can also be exported in JSON format to be ingested by the
systems being
developed by Charles River Analytics and SRI. The probabilistic
information is
encoded in two ways:

- For CRA, the kernel density estimator constructed from
the gradable adjective
  data is converted to a discrete probability table and
written to the JSON
  file.
- For SRI, a seventh-order polynomial fit to the
kernel density estimator is
  exported in the JSON file.

```{.python .input}
from delphi.export import *
export_default_initial_values(G, variables_file='variables.csv')
```

Basic Modeling Interface (BMI)
------------------------------

Some portions of
the [Basic Modeling Interface
(BMI)](http://bmi-spec.readthedocs.io/en/latest/)
specification have been
implemented in Delphi as methods of the AnalysisGraph
class, namely:

- `initialize`
- `update`
- `get_input_var_names`
-
`get_output_var_names`
- `get_time_step`
- `get_time_unit`

The `initialize` and
`update` functions are currently being called under the
hood for model
execution.

Integration with ISI/MINT
-------------------------

The BMI
interface will facilitate the coupling of Delphi models with other
models that
are instantiated in MINT workflows.

Analysis Question
-----------------

What
is the impact of increasing rainfall on food security? Let us nudge
rainfall
slightly in the positive direction

We first set
$\frac{\partial{(precipitation)}}{\partial{t}} = 0.1$

by modifying the
`variables.csv` file produced by the previous command, followed
by initializing
the model using the modified file.

```{.python .input}
s0 = pd.read_csv('variables.csv', index_col=0, header=None,
                 error_bad_lines=False)[1]
s0.loc['∂(precipitation)/∂t'] = 0.1
s0.to_csv('variables.csv')
```

```{.python .input}
s0
```

```{.python .input}
from delphi.bmi import *
initialize('variables.csv')
```

We then execute the model over three time steps to see what happens to the
latent variable food security and its corresponding indicator when precipitation
increases.

We see that in general, increasing rainfall reduces food insecurity.
But do the
results make sense? Let's 'drill down' to find out.

```{.python .input}
inspect_edge(G, 'precipitation', 'food_insecurity')
```

```{.python .input}
inspect_edge(G, 'precipitation', 'intervention')
```

```{.python .input}
inspect_edge(G, 'intervention', 'conflict')
```

```{.python .input}
inspect_edge(G, 'conflict', 'food_insecurity')
```

Looks like the results are qualitatively what we would expect given the
statements!

Discovering the unknown unknowns
--------------------------------
So far we have been working with a very limited set of 81 INDRA statements to
create our analysis graph, where each statement had one of the concepts *food
insecurity*, *precipitation*, and *conflict*. As a result, the causal
exploration was extremely limited.

What happens when we have a much larger
knowledge store to work with? What
additional causal relations can we see? To
answer these questions, we will use a
set of 2354 INDRA statements to assemble
our analysis graph, and then use some
of Delphi's graph inspection methods to
explore the causal patterns in the
graph.

```{.python .input}
full_statements = load_statements('../data/eval_stmts_grounding_filtered.pkl')
G_full = AnalysisGraph.from_statements(full_statements)
visualize(G_full)
```

Note that the usage of opacity to denote evidence strength makes the full
analysis graph - which could have been a total hairball - a bit more tractable,
and draws the analyst's eye towards the edges in the CAG that are most likely to
be important.

As before, we will merge the node `food_security` into the node
`food_insecurity`.

```{.python .input}
merge_nodes(G_full, 'food_security', 'food_insecurity', same_polarity=False)
visualize(G_full)
```

This reduces the graph complexity a bit, as expected.

Next, we visualize the
subgraph consisting of the immediate ancestors of food insecurity.

```{.python .input}
n = 'food_insecurity'
visualize(get_subgraph_for_concept(G_full, n, depth_limit=1), nodes_to_highlight=[n])
```

We can also expand our search to the next-to-immediate ancestors of food
insecurity (i.e. nodes that have a directed path to food insecurity)

```{.python .input}
visualize(get_subgraph_for_concept(G_full, n, depth_limit=2), nodes_to_highlight=[n])
```

Causal pathways between a pair of concepts
------------------------------------------

How about if you were interested in
getting the paths between two particular
concepts, say conflict and food
insecurity? The `get_subgraph_for_concept_pair`
method gets a subgraph comprised
of length *n* paths between a specified pair of
concepts, with *n* set by the
`cutoff` parameter.

```{.python .input}
concept_pair = ('conflict', 'food_insecurity')
from delphi.subgraphs import *
visualize(get_subgraph_for_concept_pair(G, *concept_pair, cutoff=1),
  nodes_to_highlight=concept_pair)
```

If we wanted to look at length-2 paths instead, we could do the following:

```{.python .input}
visualize(get_subgraph_for_concept_pair(G, *concept_pair, cutoff=2),
  nodes_to_highlight=concept_pair, rankdir='LR')
```

Pairwise causal pathways among a collection of concepts
-------------------------------------------------------

What if you were
interested in studying the interplay between a collection of
concepts? The
following command shows all the length-1 paths between food
security,
precipitation, and conflict.

```{.python .input}
visualize(get_subgraph_for_concept_pairs(G, concepts_of_interest, cutoff=1),
       nodes_to_highlight=concepts_of_interest, rankdir='LR')
```

The next command shows all the length-2 paths.

```{.python .input}
visualize(get_subgraph_for_concept_pairs(G, concepts_of_interest, cutoff=2),
          nodes_to_highlight=concepts_of_interest, rankdir='TB')
```

We can see that we have found a number of new paths by which food insecurity,
conflict, and precipitation can affect each other.

Future plans for Delphi
-----------------------

### *Reverse inference*

Right now, we are going from
latent variables to observed variables. *However*,
we also get the reverse for
free - given observations generated by predictions
from expert models (TOPOFLOW,
DSSAT, etc.), we can infer the posterior
probability of latent state variables
like food security (right now modeled as a
continuous variable, but can also be
discrete).

### *Coupled model execution*

Related to the first point, we plan
to make sure to design Delphi keeping
coupled model execution in mind, including
implementing the rest of the BMI
spec.

### *Speeding things up*

Running
sampling and inference for nontrivial ensemble world models will likely
require
a considerable amount of computation. To be able to scale with the size
of the
models, we plan to implement Delphi's backend in C++ and wrap it with the
Python-C API, thereby preserving the user-friendly Python interface.

### *FEWS
Net Indicators*

In the near future, we plan to
connect Delphi with FEWS net
indicators for food security, famine and drought.
### *Stable API*

We are
working on releasing Delphi 3.0, which will be the first version with a
stable,
well documented API that we will endeavour to maintain. See below for a
preview
of the documentation.
# Program Analysis Seedling Kickoff Demo

*University of Arizona*

*July 26, 2018*


In this demo we will show how a dynamic Bayes network (DBN) can be constructed
from the source code for a FORTRAN program. This notebook has been tested with
the version of Delphi corresponding to the commit hash shown below.

```python
!git rev-parse HEAD
```

Some preliminaries:

```python
%load_ext autoreload
%autoreload 2
import sys
import json
from importlib import import_module
from IPython.core.display import Image
import delphi.jupyter_tools as jt
from delphi.program_analysis.scopes import Scope
from delphi.program_analysis.ProgramAnalysisGraph import (ProgramAnalysisCAG,
                                                          initialize, update)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
import numpy as np
```

We start with the FORTRAN program, `crop_yield.f` shown below.

```python
jt.display(f'../data/program_analysis/crop_yield.f')
```

## Analyze the FORTRAN source code

The FORTRAN program is now analyzed by the autoTranslate program analysis pipeline:
- Analyze the FORTRAN source to produce an XML representation of the abstract
  syntax tree (AST)
- Translate the AST into a functionally equivalent Python source file
- Extract component
assignment functions (lambda functions) and variable functional relationships
within scope (DBN-JSON).

```python
%cd ../delphi/program_analysis/autoTranslate/
!./autoTranslate ../../../data/program_analysis/crop_yield.f
%cd ../../../notebooks/
```

This generates several files.  First, program analysis autoTranslate analyses
the FORTRAN code to extract the AST (abstract syntax tree), represented in xml:

```python
jt.display('../delphi/program_analysis/autoTranslate/crop_yield.xml')
```

A functionally equivalent Python source file is generated (preserving semantics
of the FORTRAN source, such as call by reference):

```python
jt.display('../delphi/program_analysis/autoTranslate/crop_yield.py')
```

Variable assignment functions are extracted into an interface of modular
executable functions, or 'lambdas'.

```python
jt.display('../delphi/program_analysis/autoTranslate/lambdas.py')
```

The Python source is then parsed into a JSON representation of the 
statements
and functions contained in the program.

```python
dbn_json = '../delphi/program_analysis/autoTranslate/pgm.json'
jt.display(dbn_json)
```

## Assemble the DBN

Delphi takes this JSON description, discovers all scopes
and assignment 
statements in it, and builds a graph that shows how 
information
flows through the program during runtime.

### Constructing the scope tree

The
scope tree is a rooted collection of nested `Scope` objects.
We currently have
two different types of scopes that our program identifies:

- A `LoopScope`
meant to track loops found in the initial program 
- A `ContainerScope` meant to
track functions found in the initial program.

The `scope_tree` defined below
will be a `Scope` object that has other `Scope` objects as its children.
###
Viewing the DBN

The final step is to construct a DBN from `scope_tree`. 
Below
we create a digraph to show the DBN as a flow of information through 
the
program at runtime. Scopes are nested and the breadth of each scope 
is shown
with a colored bounding box. Function scopes are colored in green 
while loop
scopes are shown in blue. Each scope is labeled with the scope 
specific name as
found in the JSON specification. Variables from the program 
are shown as
ellipses contained in their appropriate scopes. Variables are 
 with the name of
the variable in the program and the name of the 
scope in which the variable was
first defined. Actions, such as variable 
assignment or conditional evaluation,
are shown as black rectangles and are 
labeled similarly to variables.

This
graph has a linked structure that shows how variables from one scope 
populate
for use into child scopes. This allows us to see the entire data 
flow profile
of the initial program at runtime.

```python
root = Scope.from_json('../delphi/program_analysis/autoTranslate/pgm.json')
A = root.to_agraph()
Image(A.draw(format='png', prog='dot'), retina=True)
```

## Executing the DBN

With the DBN now assembled and represented within Delphi,
we can execute the DBN, displayed here from a CAG view of the DBN with variable
states updating as the loop is unfolded.

```python
sys.path.append('../delphi/program_analysis/autoTranslate')
import lambdas
G = ProgramAnalysisCAG.from_agraph(A, lambdas)
```

We first initialize the variables whose assignment doesn't depend on 
other
variables.

```python
initialize(G)
G.visualize(show_values = True)
```

We then update the CAG, visiting each node and updating it
based on the function
associated with that node and the predecessors of
the node. This update is
performed recursively - if the update function requires
the value of an
predecessor node that has not been computed yet, then that
predecessor node's
update function is called in turn, and so on.

```python
update(G)
G.visualize(show_values = True)
```

Upon performing another update, we can see the values of the nodes changing.

```python
update(G)
G.visualize(show_values = True)
```

This reproduces the output of the FORTRAN code, albeit with the `DAY` variable
offset by +2 as an artefact of our indexing (future work!)

In the plots below we demonstrate some of the potential for sensitivity
analysis
using this framework. In the input cell below we define a function
for plotting
three variables of interest: `RAIN`, `TOTAL_RAIN`, and `YIELD_EST`,
as a
function of `DAY`. 

In addition, we provide an option to go 'non-
deterministic', i.e. we say that
instead of `MAX_RAIN` being constant at 4, we
assert `MAX_RAIN`$\sim \mathcal{N}(4,1)$.

```python
for n in G.nodes(data=True):
    print(n)
```

```python
def make_plots(n_samples, deterministic = True):
    variables = ('RAIN', 'TOTAL_RAIN', 'YIELD_EST')
    vals = {k:[] for k in variables}
    days = {k:[] for k in variables}
    palette = sns.color_palette()
    colors = {k:palette[i] for i, k in enumerate(vals)}
    fig, axes = plt.subplots(1,len(vals), figsize=(12, 3))
    ax = {k:axes[i] for i, k in enumerate(vals)}
    
    for _ in range(n_samples):
        G = ProgramAnalysisCAG.from_agraph(A, lambdas)
        if not deterministic:
            G.nodes['MAX_RAIN']['init_fn'] = lambda: np.random.normal(4, 1)             
        initialize(G)
        for i in range(1,31):
            update(G)
            for k in vals:
                vals[k].append(G.nodes[k]['value'])
                days[k].append(G.nodes['DAY']['value']-2)

    for k in vals:
        sns.lineplot(days[k], vals[k], ax = ax[k], label=k, color=colors[k])
        ax[k].set_xlabel('DAY')
        ax[k].set_ylabel(k)

    plt.tight_layout()

```

First, we show the results of running the program deterministically.

```python
make_plots(1)
```

Note the inflection point of `YIELD_EST` that occurs when `TOTAL_RAIN` = 40:
this is the basis of the conditional expression in the original FORTRAN code.
If we turn off the deterministic behaviour, we get the following results with
fluctuations representing random sampling noise:

```python
make_plots(1, deterministic=False)
```

Once we draw more than one sample, we can estimate the uncertainty in the
outputs
caused by the stochasticity in the input `MAX_RAIN`.

```python
make_plots(10, deterministic=False)
```

## Summary

With Delphi, we can go from FORTRAN source code to a dynamic Bayes
net. The 
internal representation of Delphi has been made flexible enough to
accommodate
custom update functions which can come a variety of sources:

-
Domain experts
- Natural language processing and gradable adjective
quantifications
- Program analysis.
