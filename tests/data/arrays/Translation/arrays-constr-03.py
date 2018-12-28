from fortran_format import *
from for2py_arrays import *


def main():
    arr = Array([(1,10)])

    # values from the implied loop (11*I, I = 1,10)
    arr_constr = implied_loop_expr((lambda x: 11*x), 1, 10, 1)
    arr_subs = subscripts(arr)

    arr.set_elems(arr_subs, arr_constr)

    fmt_obj = Format(['I5'])

    for i in range(1,10+1):
        val = arr.get((i,))
        sys.stdout.write(fmt_obj.write_line([val]))

main()
