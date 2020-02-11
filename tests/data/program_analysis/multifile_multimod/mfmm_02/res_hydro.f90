      subroutine res_hydro (jres, id, ihyd, pvol_m3, evol_m3)

      use reservoir_data_module
      use reservoir_module
      use conditional_module
      use climate_module
      use time_module
      use hydrograph_module
      use water_body_module
      
      implicit none
      
      real,  intent (in) :: pvol_m3
      real,  intent (in) :: evol_m3
      integer,  intent (in) :: jres             !none      |hru number
      integer :: nstep            !none      |counter
      integer :: tstep            !none      |hru number
      integer :: iac              !none      |counter 
      integer,  intent (in) :: id               !none      |hru number
      integer :: ial              !none      |counter
      integer :: irel             !          |
      integer,  intent (in) :: ihyd             !          |
      real :: vol                 !          |
      real :: b_lo                !          |
      character(len=1) :: action  !          |
      real :: res_h               !          |

      
      !! store initial values
      vol = wbody%flo
      nstep = 1

      do tstep = 1, nstep
          
        !calc release from decision table
        do iac = 1, dtbl_res(id)%acts
          action = "n"
          do ial = 1, dtbl_res(id)%alts
            if (dtbl_res(id)%act_hit(ial) == "y" .and. dtbl_res(id)%act_outcomes(iac,ial) == "y") then
              action = "y"
              exit
            end if
          end do
          
          !condition is met - set the release rate
          if (action == "y") then
            select case (dtbl_res(id)%act(iac)%option)
            case ("rate")
              ht2%flo = dtbl_res(id)%act(iac)%const * 86400.
            case ("days")
              select case (dtbl_res(id)%act(iac)%file_pointer)
                case ("null")
                  b_lo = 0.
                case ("pvol")
                  b_lo = pvol_m3
                case ("evol")
                  b_lo = evol_m3
              end select
              ht2%flo = (wbody%flo - b_lo) / dtbl_res(id)%act(iac)%const
            case ("weir")
              ht2%flo = res_weir(ihyd)%c * res_weir(ihyd)%k * res_weir(ihyd)%w * (res_h ** 1.5)
            case ("meas")
              irel = int(dtbl_res(id)%act(iac)%const)
              select case (recall(irel)%typ)
              case (1)    !daily
                ht2%flo = recall(irel)%hd(time%day,time%yrs)%flo
              case (2)    !monthly
                ht2%flo = recall(irel)%hd(time%mo,time%yrs)%flo
              case (3)    !annual
                ht2%flo = recall(irel)%hd(1,time%yrs)%flo
              end select
            end select
          end if
        end do    ! iac

      end do    !tstep loop
      
      return
      end subroutine res_hydro