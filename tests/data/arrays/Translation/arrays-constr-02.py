from fortran_format import *
from for2py_arrays import *


def main():
    X = Array([(1,10)])
    arr = Array([(1,10)])

    X_constr = [11,22,33,44,55,66,77,88,99,110]
    X_subs = subscripts(X)
    X.set_elems(X_subs, X_constr)

    sub_arr = implied_loop_expr((lambda x:X.get(x)), 3,8,1)
    arr_constr = flatten([10,20,sub_arr, 90, 100])
    arr_subs = subscripts(arr)

    arr.set_elems(arr_subs, arr_constr)

    fmt_obj = Format(['I5'])

    for i in range(1,10+1):
        val = arr.get((i,))
        sys.stdout.write(fmt_obj.write_line([val]))

main()
