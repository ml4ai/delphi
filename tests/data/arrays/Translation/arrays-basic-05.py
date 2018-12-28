from fortran_format import *
from for2py_arrays import *

def main():
    A = Array([(-3,1),(-4,0)])

    for i in range(-3,1+1):
        for j in range(-4,0+1):
            A.set((i,j), i+j)

    fmt_obj = Format(['5(I5,X)'])

    for i in range(-3,1+1):
        sys.stdout.write(fmt_obj.write_line([A.get((i,-4)),
                                             A.get((i,-3)),
                                             A.get((i,-2)),
                                             A.get((i,-1)),
                                             A.get((i,0))]))


main()
