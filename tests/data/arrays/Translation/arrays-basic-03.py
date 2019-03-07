from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *

def main():
    arr = Array([(1,10)])
    idx = Array([(1,10)])

    for i in range(1,10+1):
        arr.set_((i,), i*i)

    for i in range(1,5+1):
        idx.set_((i,), 2*i)
        idx.set_((i+5,), 2*i-1)

    fmt_obj = Format(['I5'])

    for i in range(1,10+1):
        val = arr.get_([idx.get_((i,))])
        sys.stdout.write(fmt_obj.write_line([val]))

main()
