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
      subroutine sir(u, params)
      double precision, dimension(0:3) :: u
      double precision, dimension(0:4) :: params
      double precision S, I, R, Y, beta, gamma, iota, N, delta_t
      double precision lambda_, ifrac, rfrac, infection, recovery
      double precision randbn

      S = u(0)
      I = u(1)
      R = u(2)
      Y = u(3)
      
      beta = params(0)
      gamma = params(1)
      iota = params(2)
      N = params(3)
      delta_t = params(4)      

      lambda_ = beta * (I + iota) / N
      ifrac = 1.0 - exp(-lambda_ * delta_t)
      rfrac = 1.0 - exp(-gamma * delta_t)
      infection = randbn(S, ifrac)
      recovery = randbn(I, rfrac)

C     Set the return values
      u(0) = S - infection
      u(1) = I + infection - recovery
      u(2) = R + recovery
      u(3) = Y + infection

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
      double precision, dimension(0:tl) :: t, S, I, R, Y
      double precision, dimension(0:3) :: u0, u
      double precision, dimension(0:4) :: params, returns
      double precision tmp, pyrand

      tmp = pyrand(.true.)    ! initialize

      params(0)= 0.1
      params(1) = 0.05
      params(2) = 0.01
      params(3) = 1000.0
      params(4) = 0.1

      do j = 1, tl
         S(j) = 0.0
         I(j) = 0.0
         R(j) = 0.0
         Y(j) = 0.0
      end do

      u0(0) = 999.0
      u0(1) = 1.0
      u0(2) = 0.0
      u0(3) = 0.0
      
      S(0) = u0(0)
      I(0) = u0(1)
      R(0) = u0(2)
      Y(0) = u0(3)

      do j = 0, 3
         u(j) = u0(j)
      end do

      u = u0
      
      do j = 1, tl
         call sir(u, params)
         S(j) = u(0)
         I(j) = u(1)
         R(j) = u(2)
         Y(j) = u(3)
      end do

      call print_output(tl, S, I, R, Y)
      
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
