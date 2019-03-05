from delphi.translators.for2py.fortran_format import *
from delphi.translators.for2py.for2py_arrays import *

def main():
    arr = Array([(1,5),(1,5)])

    for i in range(1,5+1):
        for j in range(1,5+1):
            arr.set_((i,j), i+j)

    fmt_obj = Format(['5(I5,X)'])

    for i in range(1,5+1):
        sys.stdout.write(fmt_obj.write_line([arr.get_((i,1)),
                                             arr.get_((i,2)),
                                             arr.get_((i,3)),
                                             arr.get_((i,4)),
                                             arr.get_((i,5))]))


main()
