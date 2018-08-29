# pa_unit_tests

This directory contains working notes for eventual program analysis unit tests, one unit test per program idiom.

-  `unit_tests.py` contains a series of python code snippets (presumably these would be the python "intermediate representation") with brief comments in the title describing the idiom represented.  This file is really just a collection of *notes*, not something to be executed.  We should turn these into bonafide unit tests.  
- TODO: add FORTRAN versions of these (and possibly there are several different ways of expressing the same thing in FORTRAN, so it may be a many-to-one mapping to the python intermediate representation).
- `unit_tests_DBNs.{pdf/graffle}` contains my sketches of the DBN network I believe captures the structure of the corresponding program idioms.  The .pdf can be read by anyone; the .graffle is the OmniGraffle source file I'm using for creating the graphs.  
- TODO: create associated DBN-JSON for each unit test.
