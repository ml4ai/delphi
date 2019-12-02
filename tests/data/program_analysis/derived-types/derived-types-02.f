C File: derived-types-02.f
C Illustrates more complex derived types involving fields that
C are aggregates (in this case, 1-D arrays)
C 
C This code is adapted from 
C https://courses.physics.illinois.edu/phys466/sp2013/comp_info/derived.html
C
C The generated output is:
C  1   4.000   1.000   1.500   2.000
C  2   5.000   1.500   2.000   2.500
C  3   6.000   2.000   2.500   3.000

      program main
      implicit none
      
      type mytype
          integer:: i
          real, dimension(3) :: a    !  <<< derived type element that is an array
      end type mytype

      integer i, j

      type (mytype) var
      type (mytype), dimension(3) :: x    ! <<< x: array of derived type values

      var%i = 3
      
      do i = 1, var%i
          var%a(i) = var%i + i

          do j = 1, var%i
              x(i)%a(j) = (i+j)/2.0
          end do
      end do

 10   format(I3, 4(3X, F5.3))

      do i = 1, var%i
          write(*,10) i, var%a(i), x(i)%a(1), x(i)%a(2), x(i)%a(3)
      end do

      stop
      end program main
