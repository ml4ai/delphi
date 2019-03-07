from ctypes import c_int, c_float

var1: c_int = c_int(0)
var2: c_float = c_float(0.0)

var3 = 0
var4 = 0.0

print (f"Before def foo: var1 = {var1.value} and var2 = {var2.value}")
print (f"Before def foo: var3 = {var3} and var4 = {var4}")

def foo (var1: c_int, var2: c_float, var3, var4):
   var1.value = 10 
   var2.value = 20.20
   var3 = 10
   var4 = 20.20
   print (f"In def foo: var3 = {var3} and var4 = {var4}")

foo (var1, var2, var3, var4)
print (f"After def foo: var1 = {var1.value} and var2 = {var2.value}")
print (f"After def foo: var3 = {var3} and var4 = {var4}")
