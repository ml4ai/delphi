.. _AnalysisGraph:

API
===================

The AnalysisGraph is the central data structure for Delphi.
This page describes different operations that can be performed
using it.


Constructors
------------

.. currentmodule:: delphi.AnalysisGraph
.. autosummary:: 
    :toctree: generated/

    AnalysisGraph.from_statements
    AnalysisGraph.from_pickle

Subgraphs
---------
.. currentmodule:: delphi.subgraphs
.. autosummary:: 
    :toctree: generated/

    get_subgraph_for_concept
    get_subgraph_for_concept_pair
    get_subgraph_for_concept_pairs

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


.. currentmodule:: delphi.manipulation
.. autosummary:: 
    :toctree: generated/

    merge_nodes

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
.. currentmodule:: delphi.quantification
.. autosummary::
    :toctree: generated/

    map_concepts_to_indicators

Export
------
.. currentmodule:: delphi.export
.. autosummary::
    :toctree: generated/

    to_agraph


Basic Model Interface
---------------------

These methods implement the `Basic Modeling Interface (BMI) <http://bmi-spec.readthedocs.io/en/latest/>`_.


Model Control
^^^^^^^^^^^^^

.. currentmodule:: delphi.bmi
.. autosummary::
    :toctree: generated/

    initialize
    update
    update_until

Model Information
^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    get_input_var_names
    get_output_var_names
    get_component_name


Time functions
^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    get_time_step
    get_time_units
    get_current_time
