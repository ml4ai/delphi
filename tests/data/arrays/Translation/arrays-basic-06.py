from fortran_format import *
from for2py_arrays import *

def main():
    A = Array([(-3,1),(1,5),(10,14)])

    for i in range(-3,1+1):
        for j in range(1,5+1):
            for k in range(10,14+1):
                A.set((i,j,k), i+j+k)

    fmt_obj = Format(['5(I5,X)'])

    for i in range(-3,1+1):
        for j in range(1,5+1):
            sys.stdout.write(fmt_obj.write_line([A.get((i,j,10)),
                                             A.get((i,j,11)),
                                             A.get((i,j,12)),
                                             A.get((i,j,13)),
                                             A.get((i,j,14))]))


main()
