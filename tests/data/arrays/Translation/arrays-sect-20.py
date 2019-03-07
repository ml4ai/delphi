from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *

def main():
    A = Array([(1,5),(1,5)])
    K = Array([(1,3)])

    K.set_elems(array_subscripts(K), array_values([1,3,5]))

    for i in range(1,5+1):
        for j in range(1,5+1):
            A.set_((i,j), 11*(i+j))          # A(i,j) = 11*(i+j)


    fmt_obj_10 = Format(['5(I5)'])
    fmt_obj_11 = Format(['""'])

    for i in range(1,5+1):
        sys.stdout.write(fmt_obj_10.write_line([A.get_((i,1)), A.get_((i,2)), \
                                                A.get_((i,3)), A.get_((i,4)), \
                                                A.get_((i,5))]))
    sys.stdout.write(fmt_obj_11.write_line([]))

    A_subs = idx2subs([array_values(2), array_values(K)])    # A(2, K)
    A.set_elems(A_subs, 999)

    for i in range(1,5+1):
        sys.stdout.write(fmt_obj_10.write_line([A.get_((i,1)), A.get_((i,2)), \
                                                A.get_((i,3)), A.get_((i,4)), \
                                                A.get_((i,5))]))
    sys.stdout.write(fmt_obj_11.write_line([]))

    for i in range(1,3+1):
        K.set_(i, int((K.get_(i)+3)/2))

    A_subs = idx2subs([array_values(4), array_values(K)])    # A(2, K)
    A.set_elems(A_subs, -1)
    
    for i in range(1,5+1):
        sys.stdout.write(fmt_obj_10.write_line([A.get_((i,1)), A.get_((i,2)), \
                                                A.get_((i,3)), A.get_((i,4)), \
                                                A.get_((i,5))]))


main()
