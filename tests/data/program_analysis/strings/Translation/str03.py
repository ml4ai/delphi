from strings import *

def str01():
    str1 = String(5, "abcdefghijklm")
    str2 = String(5, "abcdefghijklm")
    str3 = String(5, "ab")

    print(f'str1: len = {len(str1)}, value = "{str1._val}"')
    print(f'str2: len = {len(str2)}, value = "{str2._val}"')
    print(f'str3: len = {len(str3)}, value = "{str3._val}"')

str01()
