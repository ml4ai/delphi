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


Quantification
--------------
.. currentmodule:: delphi.AnalysisGraph
.. autosummary::
    :toctree: generated/

    AnalysisGraph.map_concepts_to_indicators
    AnalysisGraph.assemble_transition_model_from_gradable_adjectives

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
