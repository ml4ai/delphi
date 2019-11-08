from delphi.translators.for2py.format import *
from delphi.translators.for2py.strings import *

def main():
    str1 = String(10, "abcdef")

    fmt_10 = Format(['A', '": len = "', 'I2', '"; value = \""', 'A', '"\""'])
    write_str = fmt_10.write_line(["str1", len(str1), str1])
    sys.stdout.write(write_str)

main()
