      PROGRAM MAIN

      IMPLICIT NONE 

      integer, parameter :: arrsize=100000,init=0
      real,parameter :: rinit=0.
      real :: r1,r2,r3,array1(2,2),array2(arrsize)
      real(kind(1.d0)) :: r4,r5
      complex :: q
      integer :: l,b,o,z,array3(10)
      data r1,r2,r3 /1.,2.,3./, array1 /1.,2.,3.,4./
      data r4 /1.23456789012345d0/ ! correct initialization
      data r5 /1.23456789012345/   ! loses precision
      data array2 /arrsize*rinit/,q /(0.,0.)/
      data (array3(l),l=1,10) /10*init/
      data b /B'01101000100010111110100001111010'/
      data o /O'15042764172'/
      data z /Z'688be87a'/
      
      write(*,*) r4,r5

      end program main
