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


      subroutine sir_gillespie(S, I, R, beta, gamma)
        implicit none
        double precision S, I, R, beta, gamma
        double precision rateInfect, rateRecover, totalRates
        rateInfect = beta * S * I / (S + I + R)
        rateRecover = gamma * I
        totalRates = rateInfect + rateRecover

        if (0.5 < (rateInfect/totalRates)) then
            ! Delta for infection
            S = S - 1
            I = I + 1
        else
            ! Delta for recovery
            I = I - 1
            R = R + 1
        endif

      end subroutine sir_gillespie
