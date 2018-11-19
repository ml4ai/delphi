C A program to test the for2py-pp pre-processing code.  This program has
C several continuation lines.
C
C NOTE: The output of for2py-pp.py for this input file contains lines that 
C exceed fixed-form Fortran's line-length limits.  To compile the Fortran
C ode output by for2py-pp.py, use the command
C
C       gfortran -ffree-form -ffree-line-length-none <file-name>

        PROGRAM MAIN

        IMPLICIT NONE 

        REAL E,Fc,Lai, nb,N,PT,Pg, di,PAR
        REAL rm,dwf,int, TMAX,TMIN, p1, sla
        REAL PD,EMP1,EMP2,Lfmax,dwc, TMN
        REAL dwr,dw,dn,w,wc,wr,wf,tb,intot, dLAI, FL
        INTEGER DOY, endsim, COUNT
        CHARACTER*10 DYN

        REAL SWFAC1, SWFAC2  

        OPEN (2,FILE='INFILE.INP',STATUS='UNKNOWN')
        
        READ(2,10) Lfmax, EMP2,EMP1,PD,nb,rm,fc,tb,intot,n,lai,w,wr,wc
     &     ,p1,sla
   10   FORMAT(17(1X,F7.4))
        CLOSE(2)

        WRITE(*,11)
        WRITE(*,12)
   11   FORMAT('Results of plant growth simulation: ')
   12   FORMAT(/
     &/,'                Accum',
     &/,'       Number    Temp                                    Leaf',
     &/,'  Day      of  during   Plant  Canopy    Root   Fruit    Area',
     &/,'   of    Leaf  Reprod  Weight  Weight  Weight  weight   Index',
     &/,' Year   Nodes    (oC)  (g/m2)  (g/m2)  (g/m2)  (g/m2) (m2/m2)',
     &/,' ----  ------  ------  ------  ------  ------  ------  ------')

        WRITE(*,11)
        WRITE(*,12)

        WRITE(*,10) Lfmax, EMP2,EMP1,PD,nb,rm,fc,tb,intot,n,lai,w,wr,wc
     &     ,p1,sla

        STOP
        END PROGRAM MAIN
