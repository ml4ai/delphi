This directory contains development code for integration with the Ibex library.

To use the patch file `patch-vibes-packaging.diff`, copy it to the root of the
VIBES repo and do 

    patch < patch-vibes-packaging.diff

Then reinstall VIBES from source according to their instructions.


Once you have done this, you should (hopefully) be able to use the Makefile to
compile the `ibex_fileout_adarsh` executable.
