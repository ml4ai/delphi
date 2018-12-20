C File: arrays-basic-07.f
C Illustrates: transposition of a 2-D matrix

	program main
	implicit none

	integer, dimension(3,5) :: A
	integer, dimension(5,3) :: B
	integer :: i, j

C Initialize matrix A
	do i = 1,3
	    do j = 1,5
	        A(i,j) = (i*j)+(i+j)
	    end do
	end do

C Transpose A into B
	do i = 1,3
	    do j = 1,5
	        B(j,i) = A(i,j)
	    end do
	end do

C Print out the results
 10     format('A:', 5(X,I4))
 11     format('')
 12     format('B:', 3(X,I4))

	do i = 1,3
	    write(*,10) A(i,1), A(i,2), A(i,3), A(i,4), A(i,5)
	end do

	write(*,11)

	do i = 1,5
	    write(*,12) B(i,1), B(i,2), B(i,3)
	end do

	stop
	end program main
