from format import *
from strings import *

def main():
    str1 = String(10)
    str2 = String(5)

    str1.set_("abcdefgh")
    str2.set_(str1.get_substr(3,8))
    str1.set_substr(2, 4, str2)

    fmt_10 = Format(['A', '"; "', 'A'])
    write_str = fmt_10.write_line([str1, str2])
    sys.stdout.write(write_str)

main()
