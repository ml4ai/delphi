      subroutine model(S, I, R, gamma, rho, totalRates)

      integer S, I, R
      double precision, parameter :: beta = rho * gamma
      double precision rateInfect, rateRecover, totalRates
      rateInfect = beta * S * I / (S + I + R)
      rateRecover = gamma * I
      totalRates = rateInfect + rateRecover


      if (rand() < (rateInfect/totalRates)) then

          S = S - 1
          I = I + 1


      else

          I = I - 1
          R = R + 1
      endif
      end subroutine model
      subroutine solver(S, I, R, gamma, rho)
        integer S, I, R
        integer, parameter :: Tmax = 100, total_runs = 1000
        double precision, parameter :: beta = rho * gamma
        double precision, dimension(0:Tmax) :: MeanS, MeanI, MeanR
        double precision, dimension(0:Tmax) :: VarS, VarI, VarR
        integer, dimension(0:Tmax) :: samples
        integer i, runs, sample_idx, sample, n_S, n_I, n_R
        double precision totalRates, dt, t
        do i = 0, Tmax
           MeanS(i) = 0
           MeanI(i) = 0.0
           MeanR(i) = 0.0
           VarS(i) = 0.0
           VarI(i) = 0.0
           VarR(i) = 0.0
           samples(i) = i
        end do
        do runs = 0, total_runs-1
           t = 0.0

           sample_idx = 0
           do while (t .le. Tmax .and. I .gt. 0)
              n_S = S
              n_I = I
              n_R = R
              call model(S, I, R, gamma, rho, totalRates)
              dt = -log(1.0-rand())/totalRates
              t = t + dt


              do while (sample_idx < Tmax .and. t > samples(sample_idx))
                 sample = samples(sample_idx)
                 runs1 = runs+1
                 MeanS(samp) = MeanS(samp)+(n_S-MeanS(samp))/(runs1)
                 VarS(samp) = VarS(samp) + runs/(runs1) * (n_S-MeanS(samp))*(n_S-MeanS(samp))

                 MeanI(samp) = MeanI(samp)+(n_I-MeanI(samp))/(runs1)
                 VarI(samp) = VarI(samp) + runs/(runs1) * (n_I-MeanI(samp))*(n_I-MeanI(samp))

                 MeanR(samp) = MeanR(samp) + (n_R - MeanR(samp))/(runs1)
                 VarR(samp) = VarR(samp) + runs/(runs1) * (n_R-MeanR(samp))*(n_R-MeanR(samp))

                 sample_idx = sample_idx+1
              end do
           end do

           do while (sample_idx < Tmax)
              sample = samples(sample_idx)
              runs1 = runs+1
              MeanS(samp) = MeanS(samp)+(n_S-MeanS(samp))/(runs1)
              VarS(samp) = VarS(samp) + runs/(runs1) * (n_S-MeanS(samp))*(n_S-MeanS(samp))

              MeanI(samp) = MeanI(samp)+(n_I-MeanI(samp))/(runs1)
              VarI(samp) = VarI(samp) + runs/(runs1) * (n_I-MeanI(samp))*(n_I-MeanI(samp))

              MeanR(samp) = MeanR(samp) + (n_R - MeanR(samp))/(runs1)
              VarR(samp) = VarR(samp) + runs/(runs1) * (n_R-MeanR(samp))*(n_R-MeanR(samp))

              sample_idx = sample_idx + 1
           end do
        end do
      end subroutine solver
      program main
      integer, parameter :: S = 500, I = 10, R = 0, Tmax = 100
      double precision, parameter :: gamma = 1.0/3.0, rho = 2.0
      call solver(S, I, R, gamma, rho)
      end program main
