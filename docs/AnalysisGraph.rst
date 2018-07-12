.. _AnalysisGraph:

AnalysisGraph Class
===================

The AnalysisGraph is the central data structure for Delphi.

.. currentmodule:: delphi.AnalysisGraph


Constructor
-----------

.. automethod:: AnalysisGraph.__init__
.. automethod:: AnalysisGraph.from_statements
.. automethod:: AnalysisGraph.from_pickle

Subgraphs
---------
.. autosummary:: 
    :toctree: generated/

    AnalysisGraph.get_subgraph_for_concept
    AnalysisGraph.get_subgraph_for_concept_pair
    AnalysisGraph.get_subgraph_for_concept_pairs

Quantification
--------------
.. autosummary::
    :toctree: generated/

    AnalysisGraph.map_concepts_to_indicators
    AnalysisGraph.infer_transition_model

Export
------
.. autosummary::
    :toctree: generated/

    AnalysisGraph.export


Basic Model Interface
---------------------

These methods implement the `Basic Modeling Interface (BMI) <http://bmi-spec.readthedocs.io/en/latest/>`_.

.. autosummary::
    :toctree: generated/

    AnalysisGraph.initialize
    AnalysisGraph.update
    AnalysisGraph.get_input_var_names
    AnalysisGraph.get_output_var_names
    AnalysisGraph.get_time_step
    AnalysisGraph.get_time_units
    AnalysisGraph.get_current_time

Inspection
----------
.. autosummary::
    :toctree: generated/

    AnalysisGraph.visualize
    AnalysisGraph.inspect_edge
