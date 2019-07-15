C     Fortranification of AMIDOL's SIR-Gillespie.py

      program SIR_Gillespie
      implicit none

C     float: Maximum time for the simulation
      integer, parameter :: T = 100
      integer nSamples, sample_idx
      double precision S0 I0, R0, t, totalRuns, gamma, rho, beta
      double precision n_S, n_I, n_R

      double precision, dimension(0:T) :: MeanS MeanI, MeanR
      double precision, dimension(0:T) :: VarS, VarI, VarR
      double precision, dimension(0:T) :: samples

C     int: Initial value of S
      S0 = 500.0
C     int: Initial value of I
      I0 = 10.0
C     int: Initial value of R
      R0 = 0.0

C     float: Initial time for the simulation
      t = 0.0

C     int: Total number of trajectories to generate for the analysis
      totalRuns = 1000

C     float: Rate of recovery from an infection
      gamma = 1.0 / 3.0

C     float: Basic reproduction Number
      rho = 2.0

C     float: Rate of infection
      beta = rho * gamma

      MeanS = 0.0
      MeanI = 0.0
      MeanR = 0.0

      VarS = 0.0
      VarI = 0.0
      VarR = 0.0

      nSamples = 0

      do runs = 0, totalRuns
        do i = 1, T
          samples(i) = i
        end do
C       Restart the event clock
        t = 0.0

C       Set the initial conditions of S, I, and R
C       int: n_S - current number of Susceptible
C       int: n_I - current number of Infected
C       int: n_R - current number of Recovered
        n_S = S0
        n_I = I0
        n_R = R0

C       Main Gillespie Loop
        sample_idx = 1
        do while ((t < T) .AND. (n_I > 0.0))
C         float: Current state dependent rate of infection
          rateInfect = beta * n_S * n_I / (n_S + n_I + n_R)
C         float: Current state dependent rate of recovery
          rateRecover = gamma * float(n_I)
C         Sum of total rates; taking advantage of Markovian identities to improve
C         performance.
          totalRates = rateInfect + rateRecover

C         float: next inter-event time
          dt = -log(1.0 - random.uniform(0.0, 1.0)) / totalRates

C         Advance the system clock
          t = t + dt
          do while (sample_idx < T .AND. t > samples(sample_idx))
            sample = samples(sample_idx)
C           Welford's one pass algorithm for mean and variance
            MeanS[sample] += (n_S - MeanS(sample)) / (runs + 1)
            MeanI[sample] += (n_I - MeanI(sample)) / (runs + 1)
            MeanR[sample] += (n_R - MeanR(sample)) / (runs + 1)
            VarS[sample] += runs / (runs + 1) * (n_S - MeanS(sample)) * (n_S - MeanS(sample))
            VarI[sample] += runs / (runs + 1) * (n_I - MeanI(sample)) * (n_I - MeanI(sample))
            VarR[sample] += runs / (runs + 1) * (n_R - MeanR(sample)) * (n_R - MeanR(sample))
            sample_idx += 1
          end do

C        Determine which event fired.  With probability rateInfect/totalRates
C        the next event is infection.
         if (rand() < (rateInfect / totalRates)) then
C          Delta for infection
           n_S = n_S - 1
           n_I = n_I + 1
C        Determine the event fired.  With probability rateRecover/totalRates
C        the next event is recovery.
         else
C          Delta for recovery
           n_I = n_I - 1
           n_R = n_R + 1
          end if
        end do

        do while (sample_idx < T)
          sample = samples(sample_idx)
C         Welford's one pass algorithm for mean and variance
          MeanS[sample] += (n_S - MeanS(sample)) / (runs + 1)
          MeanI[sample] += (n_I - MeanI(sample)) / (runs + 1)
          MeanR[sample] += (n_R - MeanR(sample)) / (runs + 1)
          VarS[sample] += runs / (runs + 1) * (n_S - MeanS(sample)) * (n_S - MeanS(sample))
          VarI[sample] += runs / (runs + 1) * (n_I - MeanI(sample)) * (n_I - MeanI(sample))
          VarR[sample] += runs / (runs + 1) * (n_R - MeanR(sample)) * (n_R - MeanR(sample))
          sample_idx += 1
        end do
      end do

      do i = 1, T
        VarS = (VarS(i) / totalRuns)
        VarI = (VarI(i) / totalRuns)
        VarR = (VarR(i) / totalRuns)
      end do

C     NOTE: Choosing to not add the plotting components
C     TODO: @PA team choose how to output the results
      stop
      end program SIR_Gillespie
