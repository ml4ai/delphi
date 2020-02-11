      subroutine flow_dur_curve

      use time_module
      use hydrograph_module
      
      implicit none  

      real, dimension(time%nbyr) :: val      !             |
      real :: sum                         !             |
      integer :: iyr                      !none         |number of years
      integer :: next                     !             |
      integer :: npts                     !             |
      integer :: ipts                     !             |
      integer :: iprv                     !             |
      integer :: mle                      !             |
      integer :: nprob                    !             |
      integer :: iday                     !             |
      integer :: mfe                      !             |
      integer :: iyr_ch                   !             | 
      
      
      val = 0.
        !set linked list for daily flow duration curves
        ob(icmd)%fdc_ll(time%day)%val = ob(icmd)%hd(1)%flo
        next = ob(icmd)%fdc%mfe
        npts = time%day - 1
        do ipts = 1, npts
          if (ob(icmd)%fdc_ll(time%day)%val <= ob(icmd)%fdc_ll(next)%val) then
            ob(icmd)%fdc_ll(time%day)%next = next
            if (ipts == 1) then
              ob(icmd)%fdc%mfe = time%day
            else
              ob(icmd)%fdc_ll(iprv)%next = time%day
            end if
            exit
          end if
          iprv = next
          next = ob(icmd)%fdc_ll(next)%next
        end do
        if (npts > 0 .and. ipts == npts + 1) then
          mle = ob(icmd)%fdc%mle
          ob(icmd)%fdc_ll(mle)%next = time%day
          ob(icmd)%fdc%mle = time%day
        end if
        !set linked list for daily flow duration curves
        
        !save flow duration curve probabilities for the year
        if (time%end_yr == 1) then
          sum = 0.
          nprob = 1
          next = ob(icmd)%fdc%mfe
          do iday = 1, time%day
            if (iday == fdc_p(nprob)) then
              select case (nprob)
              case (1)  ! 5% prob
                ob(icmd)%fdc%p(time%yrs)%p5 = ob(icmd)%fdc_ll(next)%val
              case (2)  ! 10% prob
                ob(icmd)%fdc%p(time%yrs)%p10 = ob(icmd)%fdc_ll(next)%val
              case (3)  ! 25% prob
                ob(icmd)%fdc%p(time%yrs)%p25 = ob(icmd)%fdc_ll(next)%val
              case (4)  ! 50% prob
                ob(icmd)%fdc%p(time%yrs)%p50 = ob(icmd)%fdc_ll(next)%val
              case (5)  ! 75% prob
                ob(icmd)%fdc%p(time%yrs)%p75 = ob(icmd)%fdc_ll(next)%val
              case (6)  ! 90% prob
                ob(icmd)%fdc%p(time%yrs)%p90 = ob(icmd)%fdc_ll(next)%val
              case (7)  ! 95% prob
                ob(icmd)%fdc%p(time%yrs)%p95 = ob(icmd)%fdc_ll(next)%val
              end select
              nprob = nprob + 1
            end if
            sum = sum + ob(icmd)%fdc_ll(next)%val
            next = ob(icmd)%fdc_ll(next)%next
          end do
          ob(icmd)%fdc%p(time%yrs)%mean = sum / float(iday + 1)
          mfe = ob(icmd)%fdc%mfe
          mle = ob(icmd)%fdc%mle
          ob(icmd)%fdc%p(time%yrs)%min = ob(icmd)%fdc_ll(mfe)%val
          ob(icmd)%fdc%p(time%yrs)%max = ob(icmd)%fdc_ll(mle)%val
          !write ob(icmd)%fdc%p before we reinitialize
          ob(icmd)%fdc%mfe = 1
          ob(icmd)%fdc%mle = 1
        end if
        
        !save flow duration curve probabilities for the year
        
        !master duration curve from annual curves (median)
        if (time%end_sim == 1) then
          do nprob = 1, 7
            ob(icmd)%fdc%mfe = 1
            ob(icmd)%fdc%mle = 1
            select case (nprob)
              case (1)  ! 5% prob
                do iyr = 1, time%nbyr
                  val(iyr) = ob(icmd)%fdc%p(iyr)%p5
                end do
              case (2)  ! 10% prob
                do iyr = 1, time%nbyr
                  val(iyr) = ob(icmd)%fdc%p(iyr)%p10
                end do
              case (3)  ! 25% prob
                do iyr = 1, time%nbyr
                  val(iyr) = ob(icmd)%fdc%p(iyr)%p25
                end do
              case (4)  ! 50% prob
                do iyr = 1, time%nbyr
                  val(iyr) = ob(icmd)%fdc%p(iyr)%p50
                end do
              case (5)  ! 75% prob
                do iyr = 1, time%nbyr
                  val(iyr) = ob(icmd)%fdc%p(iyr)%p75
                end do
              case (6)  ! 90% prob
                do iyr = 1, time%nbyr
                  val(iyr) = ob(icmd)%fdc%p(iyr)%p90
                end do
              case (7)  ! 95% prob
                do iyr = 1, time%nbyr
                  val(iyr) = ob(icmd)%fdc%p(iyr)%p95
                end do
              end select
              
          do iyr = 1, time%nbyr
            ob(icmd)%fdc_lla(iyr)%val = val(iyr)
            next = ob(icmd)%fdc%mfe
            npts = iyr - 1
            do ipts = 1, npts
              if (ob(icmd)%fdc_lla(iyr)%val <= ob(icmd)%fdc_lla(next)%val) then
                ob(icmd)%fdc_lla(iyr)%next = next
                if (ipts == 1) then
                  ob(icmd)%fdc%mfe = iyr
                else
                  ob(icmd)%fdc_lla(iprv)%next = iyr
                end if
                exit
              end if
              iprv = next
              next = ob(icmd)%fdc_lla(next)%next
            end do  !ipts
            if (npts > 0 .and. ipts == npts + 1) then
              mle = ob(icmd)%fdc%mle
              ob(icmd)%fdc_lla(mle)%next = iyr
              ob(icmd)%fdc%mle = iyr
            end if
          end do    !iyr
          
          !calc mean, abs max and min
          sum = 0.
          do iyr = 1, time%nbyr
            sum = sum + ob(icmd)%fdc%p(iyr)%mean
            ob(icmd)%fdc%p_md%max = Max (ob(icmd)%fdc%p_md%max, ob(icmd)%fdc%p(iyr)%max)
            ob(icmd)%fdc%p_md%min = amin1 (ob(icmd)%fdc%p_md%min, ob(icmd)%fdc%p(iyr)%min)
          end do
          ob(icmd)%fdc%p_md%mean = sum / time%nbyr
          
          !calc probabilities
          next = ob(icmd)%fdc%mfe
          do iyr = 1, time%nbyr
            if (iyr > time%nbyr / 2) then
              select case (nprob)
              case (1)  ! 5% prob
                iyr_ch = (time%nbyr + 1.1) / 2
                if (iyr <= iyr_ch) then
                  ob(icmd)%fdc%p_md%p5 = ob(icmd)%fdc%p(next)%p5
                else
                  ob(icmd)%fdc%p_md%p5 = (ob(icmd)%fdc%p(next)%p5 + ob(icmd)%fdc%p(iprv)%p5) / 2.
                end if
                exit
              case (2)  ! 10% prob
                iyr_ch = (time%nbyr + 1.1) / 2
                if (iyr <= iyr_ch) then
                  ob(icmd)%fdc%p_md%p10 = ob(icmd)%fdc%p(next)%p10
                else
                  ob(icmd)%fdc%p_md%p10 = (ob(icmd)%fdc%p(next)%p10 + ob(icmd)%fdc%p(iprv)%p10) / 2.
                end if
                exit
              case (3)  ! 25% prob
                iyr_ch = (time%nbyr + 1.1) / 2
                if (iyr <= iyr_ch) then
                  ob(icmd)%fdc%p_md%p25 = ob(icmd)%fdc%p(next)%p25
                else
                  ob(icmd)%fdc%p_md%p25 = (ob(icmd)%fdc%p(next)%p25 + ob(icmd)%fdc%p(iprv)%p25) / 2.
                end if
                exit
              case (4)  ! 50% prob
                iyr_ch = (time%nbyr + 1.1) / 2
                if (iyr <= iyr_ch) then
                  ob(icmd)%fdc%p_md%p50 = ob(icmd)%fdc%p(next)%p50
                else
                  ob(icmd)%fdc%p_md%p50 = (ob(icmd)%fdc%p(next)%p50 + ob(icmd)%fdc%p(iprv)%p50) / 2.
                end if
                exit
              case (5)  ! 75% prob
                iyr_ch = (time%nbyr + 1.1) / 2
                if (iyr <= iyr_ch) then
                  ob(icmd)%fdc%p_md%p75 = ob(icmd)%fdc%p(next)%p75
                else
                  ob(icmd)%fdc%p_md%p75 = (ob(icmd)%fdc%p(next)%p75 + ob(icmd)%fdc%p(iprv)%p75) / 2.
                end if
                exit
              case (6)  ! 90% prob
                iyr_ch = (time%nbyr + 1.1) / 2
                if (iyr <= iyr_ch) then
                  ob(icmd)%fdc%p_md%p90 = ob(icmd)%fdc%p(next)%p90
                else
                  ob(icmd)%fdc%p_md%p90 = (ob(icmd)%fdc%p(next)%p90 + ob(icmd)%fdc%p(iprv)%p90) / 2.
                end if
                exit
              case (7)  ! 95% prob
                iyr_ch = (time%nbyr + 1.1) / 2
                if (iyr <= iyr_ch) then
                  ob(icmd)%fdc%p_md%p95 = ob(icmd)%fdc%p(next)%p95
                else
                  ob(icmd)%fdc%p_md%p95 = (ob(icmd)%fdc%p(next)%p95 + ob(icmd)%fdc%p(iprv)%p95) / 2.
                end if
                exit
              end select
            end if
            iprv = next
            next = ob(icmd)%fdc_lla(next)%next
          end do    !iyr

          end do    !nprob
         write (6000,*) ob(icmd)%typ, ob(icmd)%props, ob(icmd)%fdc%p_md
        end if
               
        !master duration curve from annual curves (median)
   
      return
      end  subroutine flow_dur_curve