      LOGICAL FUNCTION FLEXIST (FILE_NAME)
      IMPLICIT NONE

*     FORMAL_PARAMETERS:
      CHARACTER*(*) FILE_NAME

**    Local variables
      CHARACTER*132 FILE_NAME_L
      LOGICAL THERE
      INTEGER IL
      SAVE

      FILE_NAME_L = FILE_NAME
      CALL FLNAME (FILE_NAME_L)
      IL = LEN_TRIM (FILE_NAME_L)

      INQUIRE (FILE=FILE_NAME_L(1:IL),EXIST=THERE)
      FLEXIST = THERE

      RETURN
      END
