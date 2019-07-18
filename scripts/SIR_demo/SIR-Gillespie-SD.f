C     Fortranification of AMIDOL's SIR-Gillespie.py

********************************************************************************
      subroutine print_output(L)
      implicit none
      integer, parameter :: Tmax = 100
      double precision, dimension(0:Tmax) :: L
      integer i

      do i = 0, 20
         write (*, 10) L(i+0), L(i+1), L(i+2), L(i+3), L(i+4)
      end do
      write (*, 11)

 10   format (5(F10.4,X))
 11   format('----------')

      return
      end subroutine print_output

********************************************************************************
*                                                                              *
*                Pretend to be Python's ramdon-number generator                *
*                                                                              *
********************************************************************************
      double precision function pyrand(do_init)
      logical do_init
      double precision retval
      
      if (do_init .eqv. .TRUE.) then        ! initialize
         open (2, file = 'PY_RANDOM_GILLESPIE')
         retval = 0.0
      else
         read (2, 10) retval
 10      format(F20.18)
      end if

      pyrand = retval
      end function pyrand

********************************************************************************
      subroutine update_mean_var(means, variances, k, n, runs)
      integer, parameter :: Tmax = 100
      double precision, dimension(0:Tmax) :: means, variances
      integer k, n, runs
      double precision upd_mean, upd_var

      upd_mean = (n - means(k))/(runs+1)

!      write (*,10) k, means(k), n, runs, upd_mean
! 10   format("k:", I3, ", means[k] = ", F7.3, ", n = ", I3, 
!     &   ", runs =", I3, ", upd_mean: ", F7.3)

      means(k) = means(k) + upd_mean

      upd_var = runs/(runs+1) * (n-means(k))*(n-means(k))
      variances(k) = variances(k) + upd_var

      return
      end subroutine update_mean_var

********************************************************************************
      program main

      integer, parameter :: S0 = 500, I0 = 10, R0 = 0, Tmax = 100
      integer, parameter :: total_runs = 10 !1000
      double precision, parameter :: gamma = 1.0/3.0  ! Rate of recovery from an infection
      double precision, parameter :: rho = 2.0  ! Basic reproduction Number
      double precision, parameter :: beta = rho * gamma ! Rate of infection

      double precision, dimension(0:Tmax) :: MeanS, MeanI, MeanR
      double precision, dimension(0:Tmax) :: VarS, VarI, VarR
      integer, dimension(0:Tmax) :: samples
      double precision pyrand

      integer i, n_samples, runs, n_S, n_I, n_R, sample_idx, sample
      double precision t, randval
      double precision rateInfect, rateRecover, totalRates, dt

      ! DEBUGGING VALUES
      double precision logval

      do i = 0, Tmax    ! Initialize the mean and variance arrays
         MeanS(i) = 0
         MeanI(i) = 0.0
         MeanR(i) = 0.0

         VarS(i) = 0.0
         VarI(i) = 0.0
         VarR(i) = 0.0

         samples(i) = i
      end do

      n_samples = 0

      randval = pyrand(.true.)    ! initialize the random-number sequence

      do runs = 0, total_runs
         t = 0.0    ! Restart the event clock

         n_S = S0
         n_I = I0
         n_R = R0

C main Gillespie loop
         sample_idx = 0
         do while (t .le. Tmax .and. n_I .gt. 0)
            rateInfect = beta * n_S * n_I / (n_S + n_I + n_R)
            rateRecover = gamma * n_I
            totalRates = rateInfect + rateRecover

            randval = pyrand(.false.)
            logval = log(1.0-randval)
            dt = -log(1.0-randval)/totalRates  ! next inter-event time

            t = t + dt          !  Advance the system clock

            ! Calculate all measures up to the current time t using
            ! Welford's one pass algorithm 
            do while (sample_idx < Tmax .and. t > samples(sample_idx))
               sample = samples(sample_idx)
!               write(*,10) sample
! 10            format("@@@ sample =", I3)
               call update_mean_var(MeanS, VarS, sample, n_S, runs)
               call update_mean_var(MeanI, VarI, sample, n_I, runs)
               call update_mean_var(MeanR, VarR, sample, n_R, runs)
               sample_idx = sample_idx+1
             end do

            ! Determine which event fired.  With probability rateInfect/totalRates
            ! the next event is infection.

            randval = pyrand(.false.)
            if (randval < (rateInfect/totalRates)) then
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

      do i = 0, Tmax
         VarS(i) = VarS(i)/total_runs
         VarI(i) = VarI(i)/total_runs
         VarR(i) = VarR(i)/total_runs
      end do

      call print_output(MeanS)
      call print_output(MeanI)
      call print_output(MeanR)

      end program main
      
