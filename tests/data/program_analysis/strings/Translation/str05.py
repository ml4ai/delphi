from delphi.translators.for2py.format import *
from delphi.translators.for2py.strings import *

def main():
    str1 = String(10)

    str1.set_("abcdef")

    n1 = str1.f_index("bc")
    n2 = str1.f_index("xyz")
    n3 = str1.f_index("f ", ["back"])
    n4 = str1.f_index("cde", ["back"])
    n5 = str1.f_index("xyz", ["back"])

    fmt_10 = Format(['5(I3,X)'])
    write_str = fmt_10.write_line([n1, n2, n3, n4, n5])
    sys.stdout.write(write_str)

main()
