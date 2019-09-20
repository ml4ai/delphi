from dreal import *
import matplotlib.pyplot as plt
import numpy as np

x = Variable("x")
y = Variable("y")
z = Variable("z")

# f_sat = And(-10 <= x, x <= 10,
            # x**2  -x**4 +2 == 0)
f_sat = And(-2 <= x, x <= 2,
            -x**2 + 0.5*x**4 == 0)

result = CheckSatisfiability(f_sat, 0.001)
print(result)


i1 = Interval(3, 4)
i2 = Interval(4, 5)

# print(i1+i2)
# print(result[0].lb())

f_sat = And(0 <=x, x <= 1, 
        0 <= y, y <=1, (x*y - y)**2 == 0)

result = CheckSatisfiability(f_sat, 0.001)
# print(result)


count = 0
begin = -5
end = 5
interval = 100
step = (end - begin)/interval
x_lst = list()
y_lst = list()

while count < interval: 
    f_sat = And(begin + step*count <=x, x <= begin + step*(count+1), 
            begin <= y, y <=end, (x*y - y)**2 == 0)

    result = CheckSatisfiability(f_sat, 0.00001)
    x_lst.append(result[0].lb())
    y_lst.append(result[1].lb())
    # print('Inside Loop 1',result)
    count += 1

count = 0

while count < interval: 

    f_sat = And(begin <= x, x <= end,begin + step*count <=y, y <= begin + step*(count+1), (x*y - y)**2 == 0)

    result = CheckSatisfiability(f_sat, 0.00001)
    x_lst.append(result[0].lb())
    y_lst.append(result[1].lb())
    # print('Inside Loop 2', result)
    count += 1

for i in range(2):
    fig, axs = plt.subplots(nrows = 1, ncols = 2, sharex = True)
    ax = axs[0]
    ax.scatter(list(range(interval)), x_lst[i*interval:(i+1)*interval], color = 'r', label = 'x', s = 50)
    axs[i].axvspan(0, interval, facecolor = 'yellow', alpha = 0.25)
    ax.legend()
    ax = axs[1]
    ax.scatter(list(range(interval)), y_lst[i*interval:(i+1)*interval], color = 'b', label = 'y', s = 50)
    ax.legend()
    plt.show()


Mat = np.column_stack((x_lst, y_lst))
#print(Mat)

label = ['x', 'y']


fig = plt.figure()
count = 0
for i in range(2):
   #count = 0
    #fig, axs = plt.subplots(nrows = 1, ncols = 4, sharex = True)
    for j in range(2):
        if i != j:
            count +=1
            ax = fig.add_subplot(2,1,count)
            ax.scatter(Mat[i*interval:(i+1)*interval, i],Mat[i*(interval):(i+1)*interval, j], color = 'r', 
                    label = 'vary {0}'.format(label[i]), s = 20)
            #print('x:', Mat[i*interval:(i+1)*interval, i])
            #print('y:', Mat[i*interval:(i+1)*interval, j])
            ax.set_xlabel('{0}'.format(label[i]))
            ax.set_ylabel('{0}'.format(label[j]))
            #ax.legend()

plt.tight_layout()
plt.show()

