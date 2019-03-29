C     File: interface-01.f
C     This program illustrates the use of a generic interface.
C
C Compiling and executing this program:
C
C     gfortran -c interface-01.f
C     gfortran interface.f
C     ./a.out
C
C The output should be the following three lines:
C  587
C  389
C  937


      module mymod
      implicit none

      interface foo
          module procedure foo_int,
     &                     foo_real, 
     &                     foo_bool
      end interface

      contains
      
          subroutine foo_int(x, result)    ! handles calls to foo with integer x
          integer :: x, result
          result = 47 * x + 23
          return
          end subroutine foo_int

          subroutine foo_real(x, result)   ! handles calls to foo with real x
          real :: x
          integer :: result
          result = int(x) * 31 + 17
          return
          end subroutine foo_real

          subroutine foo_bool(x, result)   ! handles calls to foo with logical x
          logical :: x
          integer :: result
          if (x) then
             result = 937
          else
             result = -732
          end if
          return
          end subroutine foo_bool

      end module mymod

      program main
      use mymod

      integer :: x
      
 10   format(I5)

      call foo(12, x)          ! this call executes foo_int
      write(*,10) x

      call foo(12.0, x)        ! this call executes foo_real
      write(*,10) x

      call foo(.true., x)      ! this call executes foo_bool
      write(*,10) x

      stop
      end program main

      
