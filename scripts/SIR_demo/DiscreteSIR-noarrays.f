C     Fortranification of Paul's DiscreteSIR.py 
      
      program DiscreteSIR
      implicit none
      
      call simulate()
      
      stop
      end program DiscreteSIR

C  ******************************************************************************
C  *                                                                            *
C  *   A function for computing the updated amounts of infected and recovered   *
C  *                                                                            *
C  ******************************************************************************
      double precision function randbn(n, p)
      double precision n, p, q, s, a, r, x, u
      double precision pyrand
      integer i

      q = 1.0 - p
      s = p/q
      a = (n+1)*s
      r = exp(n*log(q))
      x = 0
      u = pyrand(.false.)
      do while (.true.)
         if (u < r) then
            randbn = x
            return
         else
            u = u - r
            x = x + 1
            r = r * ((a/x) - s)
         end if
      end do
      end function randbn

C  ******************************************************************************
C  *                                                                            *
C  *      Actual model code here, this should be interchangeable with SEIR      *
C  *                                                                            *
C  ******************************************************************************
      subroutine sir(u_0, u_1, u_2, u_3, params_0,
     &               params_1, params_2, params_3, params_4)
      double precision u_0, u_1, u_2, u_3
      double precision params_0, params_1, params_2, params_3, params_4
      double precision S, I, R, Y, beta, gamma, iota, N, delta_t
      double precision lambda_, ifrac, rfrac, infection, recovery
      double precision randbn

      S = u_0
      I = u_1
      R = u_2
      Y = u_3
      
      beta = params_0
      gamma = params_1
      iota = params_2
      N = params_3
      delta_t = params_4

      lambda_ = beta * (I + iota) / N
      ifrac = 1.0 - exp(-lambda_ * delta_t)
      rfrac = 1.0 - exp(-gamma * delta_t)
      infection = randbn(S, ifrac)
      recovery = randbn(I, rfrac)

C     Set the return values
      u_0 = S - infection
      u_1 = I + infection - recovery
      u_2 = R + recovery
      u_3 = Y + infection

      end subroutine sir

C  ******************************************************************************
C  *                                                                            *
C  * Driver functions for the simulation, should be top level container in GrFN *
C  *                                                                            *
C  ******************************************************************************
      subroutine simulate()
      integer, parameter :: tf = 200
      integer, parameter :: tl = tf/0.1
      integer j
      double precision t, S, I, R, Y
      double precision u0_0, u0_1, u0_2, u0_3, u_0, u_1, u_2, u_3
      double precision params_0, params_1, params_2, params_3, params_4
      double precision tmp, pyrand

      tmp = pyrand(.true.)    ! initialize

      params_0 = 0.1
      params_1 = 0.05
      params_2 = 0.01
      params_3 = 1000.0
      params_4 = 0.1

      u0_0  = 999.0
      u0_1  = 1.0
      u0_2  = 0.0
      u0_3  = 0.0
      
      S = u0_0 
      I = u0_1 
      R = u0_2 
      Y = u0_3 

      u_0 = u0_0
      u_1 = u0_1
      u_2 = u0_2
      u_3 = u0_3

      do j = 1, tl
         call sir(u_0, u_1, u_2, u_3,
     &             params_0, params_1, params_2, params_3, params_4)
         S = u_0 
         I = u_1 
         R = u_2 
         Y = u_3 
!         write (*,*) S
      end do

 10   format(4(X,F4.0))
      write(*, 10) S, I, R, Y

      end subroutine simulate

C  ******************************************************************************
C  *                                                                            *
C  *               Pretend to be Python's ramdon-number generator               *
C  *                                                                            *
C  ******************************************************************************
      double precision function pyrand(do_init)
      double precision retval
      logical do_init

      if (do_init .eqv. .TRUE.) then        ! initialize
         open (2, file = 'PYTHON-RANDOM_SEQ')
         retval = 0.0
      else
         read (2, 10) retval
 10      format(F20.18)
      end if

      pyrand = retval

      end function pyrand

      
      subroutine print_output(tl, S, I, R, Y)
      integer tl
      double precision, dimension(0:tl) :: S, I, R, Y
      
 10   format("[", 3(X,F4.0), " ...", 3(X,F4.0), "]")
      write (*, 10) S(1), S(2), S(3), S(tl-2), S(tl-1), S(tl)
      write (*, 10) I(1), I(2), I(3), I(tl-2), I(tl-1), I(tl)
      write (*, 10) R(1), R(2), R(3), R(tl-2), R(tl-1), R(tl)
      write (*, 10) Y(1), Y(2), Y(3), Y(tl-2), Y(tl-1), Y(tl)
      end subroutine print_output
