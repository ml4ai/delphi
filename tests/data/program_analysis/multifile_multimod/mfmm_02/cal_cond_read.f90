      subroutine cal_cond_read
      
!!    ~ ~ ~ PURPOSE ~ ~ ~
!!    this function computes new paramter value based on 
!!    user defined change

!!    ~ ~ ~ INCOMING VARIABLES ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    val_cur     |variable      |current parameter value
!!                               |the standard temperature (20 degrees C)
!!    chg         |data type     |contains information on variable change
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

!!    ~ ~ ~ OUTGOING VARIABLES ~ ~ ~
!!    name        |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    chg_par     |variable      |new parameter value
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

!!    ~ ~ ~ ~ ~ ~ END SPECIFICATIONS ~ ~ ~ ~ ~ ~
      
      use input_file_module
      use maximum_data_module
      use calibration_data_module
      use conditional_module
      
      implicit none

      integer, dimension (:), allocatable :: elem_cnt    !           |
      character (len=80) :: titldum                      !           |title of file
      character (len=80) :: header                       !           |header of file
      integer :: eof                                     !           |end of file
      integer :: imax                                    !none       |determine max number for array (imax) and total number in file
      logical :: i_exist                                 !none       |check to determine if file exists
      integer :: mchg_sched                              !none       |end of loop
      integer :: i                                       !none       |counter
      integer :: icond                                   !none       |counter
      
      imax = 0
      mchg_sched = 0
        
      !!read parameter change values for calibration
      inquire (file="conditional.upd", exist=i_exist)
      if (.not. i_exist .or. "conditional.upd" == "null") then
        allocate (upd_cond(0:0))
      else
      do
        open (107,file="conditional.upd")
        read (107,*,iostat=eof) titldum
        if (eof < 0) exit
        read (107,*,iostat=eof) mchg_sched
        if (eof < 0) exit
        
        allocate (upd_cond(0:mchg_sched))
        db_mx%cond_up = mchg_sched
        
        read (107,*,iostat=eof) header
        if (eof < 0) exit

      do i = 1, mchg_sched
        read (107,*,iostat=eof) upd_cond(i)%typ, upd_cond(i)%name, upd_cond(i)%cond
        if (eof < 0) exit

        !! crosswalk parameters with calibration parameter db
        do icond = 1, db_mx%d_tbl
          if (upd_cond(i)%cond == d_tbl%name) then
            upd_cond(i)%cond_num = icond
            exit
          end if
        end do
 
      end do
      exit
      end do
      end if      

      return
      end subroutine cal_cond_read