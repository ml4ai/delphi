      MODULE Interface_SoilPBalSum
!     Interface needed for optional arguments with SoilPBalSum
      INTERFACE
        SUBROUTINE SoilPBalSum (CONTROL)
          REAL, INTENT(IN), OPTIONAL :: AMTFER, Balance, CUMIMMOB
        END SUBROUTINE
      END INTERFACE
      END MODULE Interface_SoilPBalSum
