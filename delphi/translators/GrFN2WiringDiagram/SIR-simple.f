      subroutine sir(S, I, R, beta, gamma, dt)
        implicit none
        double precision S, I, R, beta, gamma, dt
        double precision infected, recovered

        infected = (-(beta*S*I) / (S + I + R)) * dt
        recovered = (gamma*I) * dt

        S = S - infected
        I = I + infected - recovered
        R = R + recovered
      end subroutine sir

C      program main
C      double precision, parameter :: S0 = 500, I0 = 10, R0 = 0
C      double precision, parameter :: beta = 0.5, gamma = 0.3, t = 1
C
C      call sir(S0, I0, R0, beta, gamma, t)
C      end program main
