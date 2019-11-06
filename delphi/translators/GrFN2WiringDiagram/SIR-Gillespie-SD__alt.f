C     Fortranification of AMIDOL's SIR-Gillespie.py

********************************************************************************
      subroutine update_mean_var(means, vars, k, n, runs)
      integer, parameter :: Tmax = 100
      double precision, dimension(0:Tmax) :: means, vars
      integer k, n, runs

      means(k) = means(k) + (n - means(k))/(runs+1)
      vars(k) = vars(k) + runs/(runs+1) * (n-means(k))*(n-means(k))

      return
      end subroutine update_mean_var

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
      subroutine gillespie(S0, I0, R0, MeanS, MeanI, MeanR, VarS, VarI,
     &                     VarR)

      integer, parameter :: Tmax = 100
      integer S0, I0, R0
      integer, parameter :: total_runs = 1000
      double precision, parameter :: gamma = 1.0/3.0
      double precision, parameter :: rho = 2.0
      double precision, parameter :: beta = rho * gamma !

      double precision, dimension(0:Tmax) :: MeanS, MeanI, MeanR
      double precision, dimension(0:Tmax) :: VarS, VarI, VarR
      integer, dimension(0:Tmax) :: samples

      integer i, runs, n_S, n_I, n_R, sample_idx, sample
      double precision rateInfect, rateRecover, totalRates, dt, t

      do i = 0, Tmax    ! Initialize the mean and variance arrays
         MeanS(i) = 0
         MeanI(i) = 0.0
         MeanR(i) = 0.0

         VarS(i) = 0.0
         VarI(i) = 0.0
         VarR(i) = 0.0

         samples(i) = i
      end do

      do runs = 0, total_runs-1
         t = 0.0    ! Restart the event clock

         n_S = S0
         n_I = I0
         n_R = R0

         ! main Gillespie loop
         sample_idx = 0
         do while (t .le. Tmax .and. n_I .gt. 0)
            rateInfect = beta * n_S * n_I / (n_S + n_I + n_R)
            rateRecover = gamma * n_I
            totalRates = rateInfect + rateRecover

            dt = -log(1.0-rand())/totalRates  ! next inter-event time

            t = t + dt          !  Advance the system clock

            ! Calculate all measures up to the current time t using
            ! Welford's one pass algorithm
            do while (sample_idx < Tmax .and. t > samples(sample_idx))
               sample = samples(sample_idx)
               call update_mean_var(MeanS, VarS, sample, n_S, runs)
               call update_mean_var(MeanI, VarI, sample, n_I, runs)
               call update_mean_var(MeanR, VarR, sample, n_R, runs)
               sample_idx = sample_idx+1
            end do

            ! Determine which event fired.  With probability rateInfect/totalRates
            ! the next event is infection.
            if (rand() < (rateInfect/totalRates)) then
                ! Delta for infection
                n_S = n_S - 1
                n_I = n_I + 1
            ! Determine the event fired.  With probability rateRecover/totalRates
            ! the next event is recovery.
            else
                ! Delta for recovery
                n_I = n_I - 1
                n_R = n_R + 1
            endif
         end do

         ! After all events have been processed, clean up by evaluating all remaining measures.
         do while (sample_idx < Tmax)
            sample = samples(sample_idx)
            call update_mean_var(MeanS, VarS, sample, n_S, runs)
            call update_mean_var(MeanI, VarI, sample, n_I, runs)
            call update_mean_var(MeanR, VarR, sample, n_R, runs)
            sample_idx = sample_idx + 1
         end do
      end do

      end subroutine gillespie

      program main
      integer, parameter :: S0 = 500, I0 = 10, R0 = 0, Tmax = 100
      double precision, dimension(0:Tmax) :: MeanS, MeanI, MeanR
      double precision, dimension(0:Tmax) :: VarS, VarI, VarR

      call gillespie(S0, I0, R0, MeanS, MeanI, MeanR, VarS, VarI, VarR)

!      call print_output(MeanS)
!      call print_output(MeanI)
!      call print_output(MeanR)

!      do i = 0, Tmax
!         VarS(i) = VarS(i)/total_runs
!         VarI(i) = VarI(i)/total_runs
!         VarR(i) = VarR(i)/total_runs
!      end do
!
!      call print_output(VarS)
!      call print_output(VarI)
!      call print_output(VarR)
!
      end program main
