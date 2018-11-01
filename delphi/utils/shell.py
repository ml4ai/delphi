""" Helper functions for shell operations. """

import os
from contextlib import contextmanager

def _change_directory(destination_directory):
    cwd = os.getcwd()
    os.chdir(destination_directory)
    try:
        yield
    except:
        pass
    finally:
        os.chdir(cwd)


cd = contextmanager(_change_directory)
