        program test_nint
          integer, dimension(5,5) :: arr
          integer :: i, j
          do i = 1, 5
            do j = 1, 5
              arr(i,j) = i+j     ! initialize the array
            end do
          end do
          print *, nint(arr)
        end program test_nint
