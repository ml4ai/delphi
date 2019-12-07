      program main

      implicit none

      CHARACTER x*6
      character y*10, z*4, m*1
      data x /'hello!'/, y /'padd'/, z /'overwrite'/, m /'a'/

      write(*,*) x
      write(*,*) y, m
      write(*,*) z

      end program main

