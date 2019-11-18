********************************************************************************
C     Input Variables:
C     S        Number of susceptible members
C     I        Number of infected members
C     R        Number of recovered members
C     beta     Rate of infection
C     gamma    Rate of recovery from an infection
C     dt       Next inter-event time
C
C     State Variables:
C     infected    Current state dependent rate of infection
C     recovered   Current state dependent rate of recovery
********************************************************************************
      subroutine sir(S, I, R, beta, gamma, dt)
        implicit none
        double precision S, I, R, beta, gamma, dt
        double precision infected, recovered, i_rate, N
        i_rate = -beta*S*I
        N = S+I+R
        infected =  (i_rate / N) * dt
        recovered = (gamma*I) * dt

        S = S - infected
        I = I + infected - recovered
        R = R + recovered
      end subroutine sir
