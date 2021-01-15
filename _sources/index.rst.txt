.. delphi documentation master file, created by
   sphinx-quickstart on Tue Feb  6 10:41:42 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Home
====

Modeling complex phenomena such as food insecurity requires reasoning
over multiple levels of abstraction and fully utilizing expert
knowledge about multiple disparate domains, ranging from the
environmental to the sociopolitical.

Delphi is a C++/Python library for assembling causal, dynamic, probabilistic
models from information extracted from two sources:

- *Text*: Delphi utilizes causal relations extracted using machine
  reading from text sources such as UN agency reports, news articles,
  and technical papers.
- *Software*: Delphi also incorporates functionality to extract
  abstracted representations of scientific models from code that
  implements them, and convert these into probabilistic models.

Usage
-----

- Assembling a model from text: [UNDER CONSTRUCTION]

- Assembling a model from Fortran code:

.. code-block:: python

  from delphi.GrFN.networks import GroundedFunctionNetwork

  G = GroundedFunctionNetwork.from_fortran_src("""\
        subroutine relativistic_energy(e, m, c, p)

        implicit none

        real e, m, c, p
        e = sqrt((p**2)*(c**2) + (m**2)*(c**4))

        return
        end subroutine relativistic_energy"""
  )
  A = G.to_agraph()
  A.draw("relativistic_energy_grfn.png", prog="dot")


.. figure:: relativistic_energy_grfn.png
  :alt: Executable Grounded Function Network constructed from Fortran source.
  :width: 100 %



Citing
------

If you use Delphi, please cite the following:

.. code-block:: bibtex

   @InProceedings{sharp-EtAl:2019:N19-4,
     author    = {Sharp, Rebecca  and  Pyarelal, Adarsh  and  Gyori, Benjamin
       and  Alcock, Keith  and  Laparra, Egoitz  and  Valenzuela-Esc\'{a}rcega,
       Marco A.  and  Nagesh, Ajay  and  Yadav, Vikas  and  Bachman, John  and
       Tang, Zheng  and  Lent, Heather  and  Luo, Fan  and  Paul, Mithun  and
       Bethard, Steven  and  Barnard, Kobus  and  Morrison, Clayton  and
       Surdeanu, Mihai},
     title     = {Eidos, INDRA, \& Delphi: From Free Text to Executable Causal Models},
     booktitle = {Proceedings of the 2019 Conference of the North American
     Chapter of the Association for Computational Linguistics (Demonstrations)},
     month     = {6},
     year      = {2019},
     address   = {Minneapolis, Minnesota},
     publisher = {Association for Computational Linguistics},
     pages     = {42-47},
     url       = {http://www.aclweb.org/anthology/N19-4008},
     keywords = {demo paper, causal relations, timelines, locations, information extraction},
   }

   @misc{Delphi,
       Author = {Adarsh Pyarelal and Paul Hein and Jon Stephens and Pratik
                 Bhandari and HeuiChan Lim and Saumya Debray and Clayton
                 Morrison},
       Title = {Delphi: A Framework for Assembling Causal Probabilistic 
                Models from Text and Software.},
       doi={10.5281/zenodo.1436915},
   }


Delphi builds upon `INDRA <https://indra.bio>`_ and `Eidos <https://github.com/clulab/eidos>`_.
For a detailed description of our procedure to convert text to models,
see `this document <http://vision.cs.arizona.edu/adarsh/export/Arizona_Text_to_Model_Procedure.pdf>`_.
Delphi is also part of the
`AutoMATES <https://ml4ai.github.io/automates/>`_ project.

.. toctree::
  :maxdepth: 2
  :caption: Contents:

  self
  installation
  usage
  model
  AnalysisGraph_API
  GrFN_API
  grfn_spec
  delphi_database
  CONTRIBUTING
  cpp_api/library_root
  grfn_openapi

License and Funding
-------------------

Delphi is licensed under the Apache License 2.0.

The development of Delphi was supported by the Defense Advanced Research
Projects Agency (DARPA) under the World Modelers (grant no. W911NF1810014) and
Automated Scientific Knowledge Extraction (agreement no. HR00111990011)
programs.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
