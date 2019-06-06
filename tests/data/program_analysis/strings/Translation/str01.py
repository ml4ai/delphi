from strings import *

def str01():
    str1 = String(10)
    str2 = String(10)
    str3 = String(15)

    str1.set_value("abcdef")
    str2.set_value("ijklmnop")
    str3.set_value(str1 + str2)

    print(f'str1: len = {len(str1)}, value = "{str1._val}"')
    print(f'str2: len = {len(str2)}, value = "{str2._val}"')
    print(f'str1//str2: len = {len(str1+str2)}, value = "{str1+str2}"')
    print(f'str3: len = {len(str3)}, value = "{str3._val}"')

str01()
