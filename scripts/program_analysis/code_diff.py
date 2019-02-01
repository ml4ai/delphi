import tangent

import delphi.translators.data.PETPT_lambdas as lambdas

func = lambdas.PETPT__lambda__IF_1_0
df = tangent.grad(func)
print(df)
