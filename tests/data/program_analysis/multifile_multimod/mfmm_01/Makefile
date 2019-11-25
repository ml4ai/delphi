# make file to compile and link the Fortran files

DEST = main.out
SRC = CSMVersion.for ModuleDefs.for  OSDefinitions.for main.for
MODS = CSMVersion.mod ModuleDefs.mod  OSDefinitions.mod
OFILES = $(SRC:.for=.o)

%.mod : %.for
	gfortran -c $<

%.o : %.for
	gfortran -c $<

$(DEST) : $(MODS) $(OFILES) 
	gfortran $(OFILES) -o $(DEST)

ModuleDefs.mod : CSMVersion.mod OSDefinitions.mod

.PHONY: clean
clean:
	/bin/rm -f *.BAK *~ *.mod *.o $(DEST)
