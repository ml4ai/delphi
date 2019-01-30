from fortran_format import *
from for2py_arrays import *


def main():
    arr = Array([(1,12)])
    idx = Array([(1,7)])

    idx_constr = [37, 43, 59, 67, 73, 79, 83]
    idx_subs = array_subscripts(idx)
    idx.set_elems(idx_subs, idx_constr)      

    arr_constr = flatten(implied_loop_expr((lambda x: [7*x, 11*x-1, idx.get_(x)]), 1, 7, 2))
    arr_subs = array_subscripts(arr)
    arr.set_elems(arr_subs, arr_constr)


    fmt_obj = Format(['I5'])

    for i in range(1,12+1):
        val = arr.get_(i)
        sys.stdout.write(fmt_obj.write_line([val]))

main()
