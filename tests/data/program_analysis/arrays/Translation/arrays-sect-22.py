from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *

def main():
    A = Array([(1,5),(-2,2)])
    B = Array([(1,5)])

    B.set_elems(array_subscripts(B), [-95, 1, -23, 45, 3])

    fmt_obj_10 = Format(['5(I5)'])
    fmt_obj_11 = Format(['""'])

    A.set_elems(array_subscripts(A), -1)

    for i in range(1,5+1):
        sys.stdout.write(fmt_obj_10.write_line([A.get_((i,-2)), A.get_((i,-1)), \
                                                A.get_((i,0)), A.get_((i,1)), \
                                                A.get_((i,2))]))
    sys.stdout.write(fmt_obj_11.write_line([]))

    start_0 = 2
    stop_0 = B.upper_bd(0)+1
    step_0 = 3

    start_1 = A.lower_bd(1)
    stop_1 = A.upper_bd(1)+1
    step_1 = 2
    A_subs = idx2subs([B.get_elems(range(start_0,stop_0,step_0)), range(start_1,stop_1,step_1)])

    A.set_elems(A_subs, 555)

    for i in range(1,5+1):
        sys.stdout.write(fmt_obj_10.write_line([A.get_((i,-2)), A.get_((i,-1)), \
                                                A.get_((i,0)), A.get_((i,1)), \
                                                A.get_((i,2))]))

main()
