      PROGRAM MAIN

      CHARACTER(LEN=4) :: Title = "BS"
      INTEGER :: DrMD = 0, PhD = 0, MS = 0, BS = 0, Others = 0

      SELECT CASE (Title)
        CASE ("DrMD")
          DrMD = DrMD + 1
          WRITE(*,10) "DrMD: ", DrMD
        CASE ("PhD")
          PhD = PhD + 1
          WRITE(*,10) "PhD: ", PhD
        CASE ("MS")
          MS = MS + 1
          WRITE(*,10) "MS: ", MS
        CASE ("BS")
          BS = BS + 1
          WRITE(*,10) "BS: ", BS
        CASE DEFAULT
          Others = Others + 1
          WRITE(*,10) "Others: ", Others
      END SELECT

 10   FORMAT(A, I2)

      END PROGRAM MAIN
