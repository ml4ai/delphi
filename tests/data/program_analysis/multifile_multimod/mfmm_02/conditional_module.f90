      module conditional_module
    
      implicit none
    
      integer :: ical_hyd                   !              |

      type conditions_var
        character(len=16) :: var            ! condition variable (ie volume, flow, sw, time, etc)
        character(len=16) :: ob             ! object variable (ie res, hru, canal, etc)
        integer :: ob_num                   ! object number
        character(len=16) :: lim_var        ! limit variable (ie evol, pvol, fc, ul, etc)
        character(len=2) :: lim_op          ! limit operator (*,+,-)
        real :: lim_const                   ! limit constant
      end type conditions_var
              
      type actions_var
        character(len=16) :: typ            ! type of action (ie reservoir release, irrigate, fertilize, etc)
        character(len=16) :: ob             ! object variable (ie res, hru, canal, etc)
        integer :: ob_num                   ! object number
        character(len=16) :: name           ! name of action
        character(len=16) :: option         ! action option - specific to type of action (ie for reservoir, option to
                                            ! input rate, days of drawdown, weir equation pointer, etc
        real :: const                       ! constant used for rate, days, etc
        real :: const2 = 1                  ! additional constant used for rate, days, etc
        character(len=16) :: file_pointer   ! pointer for option (ie weir equation pointer)
      end type actions_var
       
      type decision_table
        character (len=16) :: name                                      ! name of the decision table
        integer :: conds                                                ! number of conditions
        integer :: alts                                                 ! number of alternatives
        integer :: acts                                                 ! number of actions
        type (conditions_var), dimension(:), allocatable :: cond        ! conditions
        character(len=16), dimension(:,:), allocatable :: alt           ! condition alternatives
        type (actions_var), dimension(:), allocatable :: act            ! actions
        character(len=1), dimension(:,:), allocatable :: act_outcomes   ! action outcomes ("y" to perform action; "n" to not perform)
        character(len=1), dimension(:), allocatable :: act_hit          ! "y" if all condition alternatives (rules) are met; "n" if not
        integer, dimension(:), allocatable :: act_typ                   ! pointer to action type (ie plant, fert type, tillage implement, release type, etc)
        integer, dimension(:), allocatable :: act_app                   ! pointer to operation or application type (ie harvest.ops, chem_app.ops, wier shape, etc)
      end type decision_table
      type (decision_table), dimension(:), allocatable, target :: dtbl_lum
      type (decision_table), dimension(:), allocatable, target :: dtbl_res
      type (decision_table), dimension(:), allocatable, target :: dtbl_scen
      type (decision_table), dimension(:), allocatable, target :: dtbl_flo
      type (decision_table), pointer :: d_tbl
      
      end module conditional_module   