C      module PolicyMod
C        implicit none
C        type Policy
C          real beta
C          integer num_days
C        end type Policy
C      end module PolicyMod

      subroutine get_beta(intrinsic_growth_rate, gamma,
     &                    susceptible, relative_contact_rate,
     &                    beta)
        real intrinsic_growth_rate, susceptible, relative_contact_rate
        real inv_contact_rate, updated_growth_rate, gamma, beta

        inv_contact_rate = 1.0 - relative_contact_rate
        updated_growth_rate = intrinsic_growth_rate + gamma
        beta = updated_growth_rate / susceptible * inv_contact_rate
      end subroutine get_beta

      subroutine get_growth_rate(doubling_time, growth_rate)
        real doubling_time, growth_rate

        if (doubling_time .eq. 0.0) then
            growth_rate = 0.0
        else
          growth_rate = 2.0 ** (1.0 / doubling_time) - 1.0
        endif
      end subroutine get_growth_rate

      subroutine sir(s, i, r, beta, gamma, n)
        real s, i, r, n, s_n, i_n, r_n, beta, gamma, scale
        s_n = (-beta * s * i) + s
        i_n = (beta * s * i - gamma * i) + i
        r_n = gamma * i + r

        scale = n / (s_n + i_n + r_n)

        s = s_n * scale
        i = i_n * scale
        r = r_n * scale
      end subroutine sir

      subroutine sim_sir(s_n, i_n, r_n, gamma, i_day,
     &                   N_p, N_t, betas, days, T, S, E, I, R)
C        use PolicyMod
        real s_n, i_n, r_n, n, gamma, beta
        integer d, i_day, idx, d_idx, p_idx
        integer N_d, N_p, N_t, T(N_t), days(N_p)
C        type (Policy) policies(N_p)
        real S(N_t), E(N_t), I(N_t), R(N_t) betas(N_p)

        n = s_n + i_n + r_n
        d = i_day

        idx = 1
        do p_idx = 1, N_p
          beta = betas(p_idx)
          N_d = days(p_idx)
          do d_idx = 1, N_d
            T(idx) = d
            S(idx) = s_n
            I(idx) = i_n
            R(idx) = r_n
            idx = idx + 1
            call sir(s_n, i_n, r_n, beta, gamma, n)
            d = d + 1
          enddo
        enddo

        T(idx) = d
        S(idx) = s_n
        I(idx) = i_n
        R(idx) = r_n
      end subroutine sim_sir

      program main
C        use PolicyMod
        implicit none
        real s_n, i_n, r_n, beta, doubling_time, growth_rate
        integer p_idx
        integer, parameter :: i_day = 17, n_days = 20
        integer, parameter :: N_p = 3, N_t = 121, infectious_days = 14
        real, parameter :: relative_contact_rate = 0.05
        real, parameter :: gamma = 1.0 / infectious_days
        real, dimension(1:N_p) :: policy_betas
        integer, dimension(1:N_p) :: policy_days
        integer, dimension(1:N_t) :: T
        real, dimension(1:N_t) :: S, E, I, R

        s_n = 1000
        i_n = 0
        r_n = 0

        do p_idx = 1, N_p
          doubling_time = (p_idx - 1.0) * 5.0
          call get_growth_rate(doubling_time, growth_rate)
          call get_beta(growth_rate, gamma, s_n,
     &                  relative_contact_rate, beta)
          policy_betas(p_idx) = beta
          policy_days(p_idx) = n_days * p_idx
        end do

        call sim_sir(s_n, i_n, r_n, gamma, i_day,
     &               N_p, N_t, policy_betas, policy_days,
     &               T, S, E, I, R)
        print*, s_n, i_n, r_n
        print*, E
      end program main
