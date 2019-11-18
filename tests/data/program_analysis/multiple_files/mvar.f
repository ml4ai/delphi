      module mvar
         implicit none
      contains
         subroutine update_mean_var(MeanS, VarS, k, n, runs)
         integer, parameter :: Tmax = 100
         double precision, dimension(0:Tmax) :: MeanS, VarS
         integer k, n, runs

         MeanS(k) = MeanS(k) + (n - MeanS(k))/(runs+1)
         VarS(k) = VarS(k) + runs/(runs+1) * (n-MeanS(k))*(n-MeanS(k))
         end subroutine update_mean_var
      end module mvar


