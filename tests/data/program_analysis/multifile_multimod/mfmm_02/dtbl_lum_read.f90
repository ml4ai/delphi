      subroutine dtbl_lum_read
      
      use maximum_data_module
      use reservoir_data_module
      use landuse_data_module
      use mgt_operations_module
      use tillage_data_module
      use fertilizer_data_module
      use input_file_module
      use conditional_module
      use pesticide_data_module
      use constituent_mass_module
      
      implicit none
                  
      character (len=80) :: titldum   !           |title of file
      character (len=80) :: header    !           |header of file
      character (len=13) :: file
      integer :: eof                  !           |end of file
      integer :: i                    !none       |counter 
      integer :: mdtbl                !none       |ending of loop
      integer :: ic                   !none       |counter 
      integer :: ial                  !none       |counter 
      integer :: iac                  !none       !counter 
      logical :: i_exist              !none       |check to determine if file exists
      integer :: idb                  !none       |counter
      integer :: ilum                 !none       |counter
      integer :: iburn                !none       |counter
      
      mdtbl = 0
      eof = 0
      
      !! read all data from hydrol.dat
      inquire (file=in_cond%dtbl_lum, exist=i_exist)
      if (.not. i_exist .or. in_cond%dtbl_lum == "null") then
        allocate (dtbl_lum(0:0)) 
      else
        do
          open (107,file=in_cond%dtbl_lum)
          read (107,*,iostat=eof) titldum
          if (eof < 0) exit
          read (107,*,iostat=eof) mdtbl
          if (eof < 0) exit
          read (107,*,iostat=eof)
          if (eof < 0) exit
          allocate (dtbl_lum(0:mdtbl))

          do i = 1, mdtbl
            read (107,*,iostat=eof) header
            if (eof < 0) exit
            read (107,*,iostat=eof) dtbl_lum(i)%name, dtbl_lum(i)%conds, dtbl_lum(i)%alts, dtbl_lum(i)%acts
            if (eof < 0) exit
            allocate (dtbl_lum(i)%cond(dtbl_lum(i)%conds))
            allocate (dtbl_lum(i)%alt(dtbl_lum(i)%conds,dtbl_lum(i)%alts))
            allocate (dtbl_lum(i)%act(dtbl_lum(i)%acts))
            allocate (dtbl_lum(i)%act_hit(dtbl_lum(i)%alts))
            allocate (dtbl_lum(i)%act_typ(dtbl_lum(i)%acts))
            allocate (dtbl_lum(i)%act_app(dtbl_lum(i)%acts))
            allocate (dtbl_lum(i)%act_outcomes(dtbl_lum(i)%acts,dtbl_lum(i)%alts))
            
            !read conditions and condition alternatives
            read (107,*,iostat=eof) header
            if (eof < 0) exit
            do ic = 1, dtbl_lum(i)%conds
              read (107,*,iostat=eof) dtbl_lum(i)%cond(ic), (dtbl_lum(i)%alt(ic,ial), ial = 1, dtbl_lum(i)%alts)
              if (eof < 0) exit
            end do
                        
            !read actions and action outcomes
            read (107,*,iostat=eof) header
            if (eof < 0) exit
            do iac = 1, dtbl_lum(i)%acts
              read (107,*,iostat=eof) dtbl_lum(i)%act(iac), (dtbl_lum(i)%act_outcomes(iac,ial), ial = 1, dtbl_lum(i)%alts)
              if (eof < 0) exit
            end do

            !cross walk characters to get array numbers
            do iac = 1, dtbl_lum(i)%acts
                select case (dtbl_lum(i)%act(iac)%typ)
                    
                case ("harvest")
                  do idb = 1, db_mx%harvop_db
                    if (dtbl_lum(i)%act(iac)%file_pointer == harvop_db(idb)%name) then
                      dtbl_lum(i)%act_typ(iac) = idb
                      exit
                    end if
                  end do
                    
                  case ("harvest_kill")
                  do idb = 1, db_mx%harvop_db
                    if (dtbl_lum(i)%act(iac)%file_pointer == harvop_db(idb)%name) then
                      dtbl_lum(i)%act_typ(iac) = idb
                      exit
                    endif
                  end do
                
                  case ("till")
                  do idb = 1, db_mx%tillparm
                    if (dtbl_lum(i)%act(iac)%option == tilldb(idb)%tillnm) then
                      dtbl_lum(i)%act_typ(iac) = idb
                      exit
                    endif
                  end do
                
                case ("irr_demand")
                  do idb = 1, db_mx%irrop_db
                    if (dtbl_lum(i)%act(iac)%option == irrop_db(idb)%name) then
                      dtbl_lum(i)%act_typ(iac) = idb
                      exit
                    end if
                  end do
                  
                case ("fertilize")
                  !xwalk fert name with fertilizer data base
                  do idb = 1, db_mx%fertparm
                    if (dtbl_lum(i)%act(iac)%option == fertdb(idb)%fertnm) then
                      dtbl_lum(i)%act_typ(iac) = idb
                      exit
                    endif
                  end do
                  !xwalk application type with chemical application data base
                  do idb = 1, db_mx%chemapp_db
                    if (dtbl_lum(i)%act(iac)%file_pointer == chemapp_db(idb)%name) then
                      dtbl_lum(i)%act_app(iac) = idb
                      exit
                    endif
                  end do
                                        
                case ("pest_apply")
                  !xwalk fert name with fertilizer data base
                  do idb = 1, cs_db%num_pests
                    if (dtbl_lum(i)%act(iac)%option == cs_db%pests(idb)) then
                      dtbl_lum(i)%act_typ(iac) = idb
                      exit
                    endif
                  end do
                  !xwalk application type with chemical application data base
                  do idb = 1, db_mx%chemapp_db
                    if (dtbl_lum(i)%act(iac)%file_pointer == chemapp_db(idb)%name) then
                      dtbl_lum(i)%act_app(iac) = idb
                      exit
                    endif
                  end do
                                                             
                case ("graze")
                  !xwalk graze name with grazing data base (graze.ops)
                  do idb = 1, db_mx%grazeop_db
                    if (dtbl_lum(i)%act(iac)%option == grazeop_db(idb)%name) then
                      dtbl_lum(i)%act_typ(iac) = idb
                    end if
                  end do
                  
                case ("burn")
                  do iburn = 1, db_mx%fireop_db
                    if (dtbl_lum(i)%act(iac)%option == fire_db(iburn)%name) then
                      dtbl_lum(i)%act_typ(iac) = iburn
                      exit
                    end if
                  end do
                end select
                
            end do
            
          end do
          db_mx%dtbl_lum = mdtbl
          exit
        end do
      end if
      
      close (107)

      return  
      end subroutine dtbl_lum_read