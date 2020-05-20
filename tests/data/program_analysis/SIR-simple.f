********************************************************************************
!     Input Variables:
!     S        Amount of susceptible members at the current timestep
!     I        Amount of infected members at the current timestep
!     R        Amount of recovered members at the current timestep
!     beta     Rate of transmission via contact
!     gamma    Rate of recovery from infection
!     dt       Next inter-event time
!
!     State Variables:
!     infected    Increase in infected at the current timestep
!     recovered   Increase in recovered at the current timestep
********************************************************************************
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
