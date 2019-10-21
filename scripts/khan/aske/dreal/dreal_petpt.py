from dreal import *

tmax = Variable("tmax")
tmin = Variable("tmin")
xhlai = Variable("xhlai")
msalb = Variable("msalb")
srad = Variable("srad")


td = Variable("td")
albedo = Variable("albedo")
slang = Variable("slang")
eeq = Variable("eeq")

eo = Variable("eo")

td = 0.6*tmax + 0.4*tmin

if xhlai < 0.0:
    albedo = msalb
else:
    albedo = 0.23 - (0.23 - msalb)*exp(-0.75*xhlai)

slang = srad*23.923
eeq = slang*(2.04*pow(10, -4)-1.83*pow(10, -4)*albedo)*(td+29.0)
eo = eeq*1.1

if tmax > 35.0:
    eo = eeq*((tmax-35.0)*0.05 + 1.1)
elif tmax < 5.0:
    eo = eeq*0.01*exp(0.18*(tmax+20.0))

eo = max(eo, 0.0001)


f_sat = And(16.1 <= tmax, tmax <= 36.7, 0.0 <= tmin, tmin <= 23.9, 0 <= msalb, msalb <=
        1, 0.0 <= xhlai, xhlai <= 4.77, 2.45 <= srad, srad <= 27.8,
            0.0 <= eo, eo <= 20.0)

result = CheckSatisfiability(f_sat, 0.0001)
print(result)

