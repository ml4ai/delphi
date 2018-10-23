fhandle = open("data.inp", "r")

line = fhandle.readline()
data_list = line.split()

x = data_list[0]
y = data_list[1]
z = data_list[2]

print (x)
print (y)
print (z)
print (line)

