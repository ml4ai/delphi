from delphi.translators.for2py.scripts.fortran_format import *
from M_mymod import *

def main():
    x = [0]     # how does for2py initialize variables?
    fmt_10 = Format(['I5'])

    foo([12], x)
    line = fmt_10.write_line([x[0]])
    sys.stdout.write(line)

    foo([12.0], x)
    line = fmt_10.write_line([x[0]])
    sys.stdout.write(line)

    foo([True], x)
    line = fmt_10.write_line([x[0]])
    sys.stdout.write(line)

main()


    
