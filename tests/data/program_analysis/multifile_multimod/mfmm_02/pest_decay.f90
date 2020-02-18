      subroutine pest_decay
      
!!    ~ ~ ~ PURPOSE ~ ~ ~
!!    this subroutine calculates degradation of pesticide in the soil and on 
!!    the plants

!!    ~ ~ ~ INCOMING VARIABLES ~ ~ ~
!!    name          |units         |definition
!!    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
!!    decay_f(:)    |none          |exponential of the rate constant for
!!                                 |degradation of the pesticide on foliage
!!    decay_s(:)    |none          |exponential of the rate constant for
!!                                 |degradation of the pesticide in soil
!!    ihru          |none          |HRU number

      use pesticide_data_module
      use hru_module, only : hru, ihru
      use constituent_mass_module
      use soil_module
      use plant_module
      use output_ls_pesticide_module
      
      implicit none 
      
      integer :: j               !none     |hru number
      integer :: k               !none     |seqential pesticide number being simulated
      integer :: ipest_db        !none     |pesticide number from pesticide data base
      integer :: l               !none     |layer number 
      real :: pest_init          !kg/ha    |amount of pesticide present at beginning of day
      real :: pest_end           !kg/ha    |amount of pesticide present at end of day
      real :: pst_decay_s        !kg/ha    |amount of pesticide decay in soil during day

      j = ihru

      if (cs_db%num_pests == 0) return
     
      do k = 1, cs_db%num_pests
        hpestb_d(j)%pest(k)%decay_s = 0.
        hpestb_d(j)%pest(k)%decay_f = 0.
        ipest_db = cs_db%pest_num(k)
        if (ipest_db > 0) then
          pst_decay_s = 0.
          !! calculate degradation in soil
          do l = 1, soil(j)%nly
            pest_init = cs_soil(j)%ly(l)%pest(k)
            if (pest_init > 1.e-12) then
              pest_end = pest_init * pestcp(ipest_db)%decay_s
              cs_soil(j)%ly(l)%pest(k) = pest_end
              pst_decay_s = pst_decay_s + (pest_init - pest_end)
            end if
          end do
          hpestb_d(j)%pest(k)%decay_s = pst_decay_s

          !! calculate degradation on plant foliage
          pest_init = cs_pl(j)%pest(k)
          if (pest_init > 1.e-12) then
            pest_end = pest_init * pestcp(ipest_db)%decay_f
            cs_pl(j)%pest(k) = pest_end
            hpestb_d(j)%pest(k)%decay_f = pest_init - pest_end
          end if
        end if
      end do
      
      return
      end subroutine pest_decay