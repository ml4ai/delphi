.. _AnalysisGraph:

AnalysisGraph API
=================

The central data structure for Delphi is the AnalysiGraph, which
represents a directed graph that encodes information about causal
quantitative relationships between concepts that we are interested in
modeling.

The AnalysisGraph class currently inherits from the NetworkX DiGraph
object, but the underlying implementation might be switched out for a
faster one in the future.

Each node in the graph represents a *concept* - at the time of writing
this (01/31/2019), this means an entry in an ontology of high-level
concepts related to food security, like `this one`_.

Each concept is associated with a set of indicators, which correspond to
real-world, measurable quantities (and importantly, that Delphi has
access to normalized data for.). In practice, this set of indicators is
implemented as a dictionary, keyed by the name of the indicator. The
values of the dictionary are objects of the class
:class:`delphi.random_variables.Indicator`.

The methods listed on this page constitute the public API for Delphi.

.. autoclass:: AnalysisGraph

Constructors
------------

.. currentmodule:: delphi.AnalysisGraph
.. autosummary:: 
    :toctree: generated/

    AnalysisGraph.from_statements
    AnalysisGraph.from_statements_file
    AnalysisGraph.from_text
    AnalysisGraph.from_json_serialized_statements_list
    AnalysisGraph.from_json_serialized_statements_file
    AnalysisGraph.from_uncharted_json_file
    AnalysisGraph.from_uncharted_json_serialized_dict

Subgraphs
---------
.. autosummary:: 
    :toctree: generated/

    AnalysisGraph.get_subgraph_for_concept
    AnalysisGraph.get_subgraph_for_concept_pair
    AnalysisGraph.get_subgraph_for_concept_pairs

Inspection
----------

Methods for detailed inspection of the AnalysisGraph.

.. currentmodule:: delphi.inspection
.. autosummary::
    :toctree: generated/

    inspect_edge

Manipulation
------------

Methods to edit the AnalysisGraph.


.. currentmodule:: delphi.AnalysisGraph
.. autosummary:: 
    :toctree: generated/

    AnalysisGraph.merge_nodes
    AnalysisGraph.delete_node
    AnalysisGraph.delete_edge
    AnalysisGraph.delete_nodes
    AnalysisGraph.delete_edges
    AnalysisGraph.prune


Parameterization
----------------
.. currentmodule:: delphi.AnalysisGraph
.. autosummary::
    :toctree: generated/

    AnalysisGraph.map_concepts_to_indicators
    AnalysisGraph.parameterize

Sampling and Inference
----------------------
.. currentmodule:: delphi.AnalysisGraph
.. autosummary::
    :toctree: generated/

    AnalysisGraph.sample_from_prior
    AnalysisGraph.infer_transition_matrix_coefficient_from_data


Export
------
.. currentmodule:: delphi.AnalysisGraph
.. autosummary::
    :toctree: generated/

    AnalysisGraph.to_dict
    AnalysisGraph.to_agraph


Basic Model Interface
---------------------

These methods implement the `Basic Modeling Interface (BMI) <http://bmi-spec.readthedocs.io/en/latest/>`_.


Model control
^^^^^^^^^^^^^

.. currentmodule:: delphi.AnalysisGraph
.. autosummary::
    :toctree: generated/

    AnalysisGraph.initialize
    AnalysisGraph.update
    AnalysisGraph.update_until

Model information
^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    AnalysisGraph.get_input_var_names
    AnalysisGraph.get_output_var_names
    AnalysisGraph.get_component_name


Time functions
^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    AnalysisGraph.get_time_step
    AnalysisGraph.get_time_units
    AnalysisGraph.get_current_time

.. _this one: https://github.com/clulab/eidos/blob/master/src/main/resources/org/clulab/wm/eidos/english/ontologies/un_ontology.yml
