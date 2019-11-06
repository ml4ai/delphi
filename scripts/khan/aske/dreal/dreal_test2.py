from dreal import *

x = Variable("x")
y = Variable("y")
z = Variable("z")

f_sat = And(0 <= x, x <= 10, 0 <= y, y <= 10, 0 <= z, z <= 10,
            sin(x) + cos(y) == z)

f_unsat = And(3 <= x, x <= 4, 4 <= y, y <= 5, 5 <= z, z <= 6,
              sin(x) + cos(y) == z)


print(CheckSatisfiability(f_sat, 0.0001))

print(CheckSatisfiability(f_unsat, 0.0001))


objective = 2 * x * x + 6 * x + 5
constraint = And(-10 <= x, x <= 10)

result = Minimize(objective, constraint, 0.0001)

print(result)

objective1 = - x**2 + 0.5*x**4
constraint = And(-10 <= x, x <= 10)

result1 = Minimize(objective1, constraint, 0.00001)

print(result1)


objective2 =  x * x - 0.5 * x * x * x * x 
constraint = And(-10 <= x, x <= 10)

result2 = Minimize(objective2, constraint, 0.00001)

print(result2)

f_sat = And(0 <= x, x <= 2, objective1 == -0.5)
result3 = CheckSatisfiability(f_sat, 0.00001)

print(result3)
