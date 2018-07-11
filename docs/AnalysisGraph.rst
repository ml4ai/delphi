.. _AnalysisGraph:

AnalysisGraph Class
===================

The AnalysisGraph is the central data structure for Delphi.

.. currentmodule:: delphi.AnalysisGraph
.. autoclass:: AnalysisGraph


Constructor
-----------
.. autosummary:: 
    :toctree: generated/

    AnalysisGraph.__init__

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


Execution
---------
.. autosummary::
    :toctree: generated/

    AnalysisGraph.sample_transition_matrix
    AnalysisGraph.sample_sequence_of_latent_states
    AnalysisGraph.sample_sequence_of_observed_states

Inspection
----------
.. autosummary::
    :toctree: generated/

    AnalysisGraph.visualize
    AnalysisGraph.inspect_edge
