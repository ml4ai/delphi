.. _GroundedFunctionNetwork:

GroundedFunctionNetwork API
===========================

The central class for the program analysis component of Delphi is the
GroundedFunctionNetwork, which represents a bipartite graph that encodes
information about computational relationships between variables in code.

Additionally, the ForwardInfluenceBlanket class is provided to facilitate model
comparison

Constructors
------------

.. currentmodule:: delphi.GrFN.networks
.. autosummary:: 
    :toctree: generated/

    GroundedFunctionNetwork.from_fortran_src
    GroundedFunctionNetwork.from_fortran_file
    GroundedFunctionNetwork.from_python_src
    GroundedFunctionNetwork.from_python_file

Export
------
.. currentmodule:: delphi.GrFN.networks
.. autosummary::
    :toctree: generated/

    GroundedFunctionNetwork.to_agraph
    GroundedFunctionNetwork.to_CAG
    GroundedFunctionNetwork.to_call_agraph

Execution
---------
.. currentmodule:: delphi.GrFN.networks
.. autosummary::
    :toctree: generated/

    GroundedFunctionNetwork.run

Model Analysis
--------------
.. currentmodule:: delphi.GrFN.networks
.. autosummary::
    :toctree: generated/

    GroundedFunctionNetwork.to_FIB

Sensitivity Analysis
--------------------
.. currentmodule:: delphi.GrFN.networks
.. autosummary::
    :toctree: generated/

    GroundedFunctionNetwork.S2_surface

.. currentmodule:: delphi.GrFN.networks
.. autosummary::
    :toctree: generated/

    ForwardInfluenceBlanket.S2_surface
    ForwardInfluenceBlanket.run
