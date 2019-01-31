.. _AnalysisGraph:

AnalysisGraph API
=================

The AnalysisGraph is the central data structure for Delphi.
This page describes different operations that can be performed
using it.


Constructors
------------

.. currentmodule:: delphi.AnalysisGraph
.. autosummary:: 
    :toctree: generated/

    AnalysisGraph.from_statements
    AnalysisGraph.from_statements_file
    AnalysisGraph.from_pickle
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

    AnalysisGraph.assemble_transition_model_from_gradable_adjectives
    AnalysisGraph.sample_from_prior
    AnalysisGraph.infer_transition_matrix_coefficient_from_data


Export
------
.. currentmodule:: delphi
.. autosummary::
    :toctree: generated/

    export


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
