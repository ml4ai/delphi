from delphi.translators.for2py.format import *
from delphi.translators.for2py.strings import *

def main():
    str1 = String(10)
    str2 = String(5)
    str3 = String(15)

    str1.set_("abcdefgh")
    str2.set_(str1)
    str3.set_(str1)

    fmt_10 = Format(['A', '": len = "', 'I2', '"; value = \""', 'A', '"\""'])
    write_str = fmt_10.write_line(["str1", len(str1), str1])
    sys.stdout.write(write_str)
    write_str = fmt_10.write_line(["str2", len(str2), str2])
    sys.stdout.write(write_str)
    write_str = fmt_10.write_line(["str3", len(str3), str3])
    sys.stdout.write(write_str)

main()
