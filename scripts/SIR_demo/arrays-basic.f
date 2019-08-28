      program main
      implicit none

      integer, parameter :: Tmax = 100
      double precision :: n
      integer :: k
      integer :: runs
      integer i, j

      double precision, dimension(0:Tmax) :: means, meani, meanr
      double precision, dimension(0:Tmax) :: vars, vari, varr
      integer, dimension(0:Tmax) :: samples

      k = 0
      n = 2
      runs = 10

      means(k) = means(k) + (n - means(k))/(runs+1)
      vars(k) = vars(k) + runs/(runs+1) * (n-means(k))*(n-means(k))

      do i = 0, Tmax    ! Initialize the mean and variance arrays
         means(i) = 0
         meani(i) = 0.0
         meanr(i) = 0.0

         vars(i) = 0.0
         vari(i) = 0.0
         varr(i) = 0.0

         samples(i) = i
      end do

      end program main
