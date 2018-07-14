.. _AnalysisGraph:

AnalysisGraph Class
===================

The AnalysisGraph is the central data structure for Delphi.

.. currentmodule:: delphi.AnalysisGraph


Constructors
------------

.. autosummary:: 
    :toctree: generated/

    AnalysisGraph.from_statements
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

.. autosummary::
    :toctree: generated/

    AnalysisGraph.visualize
    AnalysisGraph.inspect_edge

Manipulation
------------

Methods to edit the AnalysisGraph.


.. autosummary:: 
    :toctree: generated/

    AnalysisGraph.merge_nodes

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


Model Control
^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    AnalysisGraph.initialize
    AnalysisGraph.update
    AnalysisGraph.update_until

Model Information
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


