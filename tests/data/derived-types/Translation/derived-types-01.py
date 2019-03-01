from delphi.translators.for2py.scripts.fortran_format import *
import sys

class mytype:
    def __init__(self):
        k : int = None
        v : float = None

def main():
    x = mytype()
    y = mytype()

    x.k = 12
    x.v = 3.456

    y.k = 21
    y.v = 4.567

    fmt_10 = Format(['2(I5, X, F6.3)'])

    line = fmt_10.write_line([x.k, y.v, y.k, x.v])
    sys.stdout.write(line)

main()


