C     Fortranification of AMIDOL's SIR-Gillespie.py

********************************************************************************

********************************************************************************
C     Variables:
C     beta     Rate of infection
C     gamma    Rate of recovery from an infection
C     rho      Basic reproduction Number
C
C     State Variables: S, I, R
C     S - Susceptible population
C     I - Infected population
C     R - Recovered population
C     n_S      current number of susceptible members
C     n_I      current number of infected members
C     n_R      current number of recovered members
C     S0       initial value of S
C     I0       initial value of I
C     R0       initial value of R
C     MeanS    Measures of Mean for S
C     MeanI    Measures of Mean for I
C     MeanR    Measures of Mean for R
C     VarS     Measures of Variance for S
C     VarI     Measures of Variance for I
C     VarR     Measures of Variance for R
C
C     rateInfect    Current state dependent rate of infection
C     rateRecover   Current state dependent rate of recovery
C     totalRates    Sum of total rates; taking advantage of Markovian identities
C                       to improve performance.
C
C     Tmax          Maximum time for the simulation
C     t             Initial time for the simulation
C     totalRuns     Total number of trajectories to generate for the analysis
C     dt       next inter-event time
********************************************************************************
      subroutine gillespie(S, I, R, gamma, rho)
        integer S, I, R
        integer, parameter :: Tmax = 100, total_runs = 1000
        double precision gamma, rho
        double precision, parameter :: beta = rho * gamma !
        double precision, dimension(0:Tmax) :: MeanS, MeanI, MeanR
        double precision, dimension(0:Tmax) :: VarS, VarI, VarR
        integer, dimension(0:Tmax) :: samples

        integer j, runs, n_S, n_I, n_R, sample_idx, samp, runs1
        double precision totalRates, dt, t, randval
        double precision rateInfect, rateRecover

        do j = 0, Tmax    ! Initialize the mean and variance arrays
          MeanS(j) = 0
          MeanI(j) = 0.0
          MeanR(j) = 0.0

          VarS(j) = 0.0
          VarI(j) = 0.0
          VarR(j) = 0.0

          samples(j) = j
        end do

        do runs = 0, total_runs-1
          t = 0.0    ! Restart the event clock

          ! main Gillespie loop
          sample_idx = 0
          do while (t .le. Tmax .and. I .gt. 0)
            n_S = S
            n_I = I
            n_R = R

            rateInfect = beta * S * I / (S + I + R)
            rateRecover = gamma * I
            totalRates = rateInfect + rateRecover

            ! Determine which event fired.  With probability rateInfect/totalRates
            ! the next event is infection.
            if (rand() < (rateInfect/totalRates)) then
              ! Delta for infection
              S = S - 1
              I = I + 1
            ! Determine the event fired.  With probability rateRecover/totalRates
            ! the next event is recovery.
            else
              ! Delta for recovery
              I = I - 1
              R = R + 1
            endif

            dt = -log(1.0-rand())/totalRates  ! next inter-event time
            t = t + dt          !  Advance the system clock

            ! Calculate all measures up to the current time t using
            ! Welford's one pass algorithm
            do while (sample_idx < Tmax .and. t > samples(sample_idx))
              samp = samples(sample_idx)

              runs1 = runs+1
              MeanS(samp) = MeanS(samp)+(n_S-MeanS(samp))/(runs1)
              VarS(samp) = VarS(samp) + runs/(runs1) *
     &                        (n_S-MeanS(samp))*(n_S-MeanS(samp))

              MeanI(samp) = MeanI(samp)+(n_I-MeanI(samp))/(runs1)
              VarI(samp) = VarI(samp) + runs/(runs1) *
     &                         (n_I-MeanI(samp))*(n_I-MeanI(samp))

              MeanR(samp) = MeanR(samp) + (n_R - MeanR(samp))/(runs1)
              VarR(samp) = VarR(samp) + runs/(runs1) *
     &                         (n_R-MeanR(samp))*(n_R-MeanR(samp))

              sample_idx = sample_idx+1
            end do
          end do

          ! After all events have been processed, clean up by evaluating all remaining measures.
          do while (sample_idx < Tmax)
            samp = samples(sample_idx)

            runs1 = runs+1
            MeanS(samp) = MeanS(samp)+(n_S-MeanS(samp))/(runs1)
            VarS(samp) = VarS(samp) + runs/(runs1) *
     &                        (n_S-MeanS(samp))*(n_S-MeanS(samp))

            MeanI(samp) = MeanI(samp)+(n_I-MeanI(samp))/(runs1)
            VarI(samp) = VarI(samp) + runs/(runs1) *
     &                         (n_I-MeanI(samp))*(n_I-MeanI(samp))

            MeanR(samp) = MeanR(samp) + (n_R - MeanR(samp))/(runs1)
            VarR(samp) = VarR(samp) + runs/(runs1) *
     &                         (n_R-MeanR(samp))*(n_R-MeanR(samp))

            sample_idx = sample_idx + 1
          end do
        end do
      end subroutine gillespie

      program main
      integer, parameter :: S = 500, I = 10, R = 0, Tmax = 100
      double precision, parameter :: gamma = 1.0/3.0, rho = 2.0

      call gillespie(S, I, R, gamma, rho)
      end program main
