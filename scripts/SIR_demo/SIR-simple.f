      subroutine sir(S, I, R, b, g, dt)
        implicit none
        double precision S, I, R, b, g, dt
        double precision N, dS, dI, dR

        N = S + I + R

        dS = (-(b*S*I) / N) * dt
        dI = (((b*S*I) / N) - g*I) * dt
        dR = (g*I) * dt

        S = S + dS
        I = I + dI
        R = R + dR
      end subroutine sir

C      program main
C      double precision, parameter :: S0 = 500, I0 = 10, R0 = 0
C      double precision, parameter :: beta = 0.5, gamma = 0.3, t = 1
C
C      call sir(S0, I0, R0, beta, gamma, t)
C      end program main
