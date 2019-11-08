from delphi.translators.for2py.format import *
from delphi.translators.for2py.strings import *

def main():
    str1 = String(10)
    str2 = String(5)

    str1.set_("   abc  def")
    str2.set_(str1.get_substr(3,7))

    fmt_10 = Format(['">>>"', 'A', '"<<<"'])
    write_str = fmt_10.write_line([str1])
    sys.stdout.write(write_str)

    write_str = fmt_10.write_line([str2])
    sys.stdout.write(write_str)

    write_str = fmt_10.write_line([str2.adjustl()])
    sys.stdout.write(write_str)

    write_str = fmt_10.write_line([str2.adjustr()])
    sys.stdout.write(write_str)

main()
