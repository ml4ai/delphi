# make file to compile and link the Fortran files

DEST = dscsm047
SRC_FOR = CSMVersion.for OSDefinitions.for ModuleDefs.for CSM.for UTILS.for \
	READS.for PATH.for DATES.for input_sub.for ERROR.for LAND.for IPIBS.for \
	SOIL.for weathr.for Warning.for OPHEAD.for OPWEATH.for IPWTH_alt.for \
	HMET.for Info.for WTHMOD.for SPAM.for ETPHOT.for OPETPHOT.for ETPHR.for \
	OpStemp.for CO2VAL.for SOLAR.for 

SRC_F90 = 

MODS_FOR = CSMVersion.for OSDefinitions.for ModuleDefs.for OPHEAD.for 

MODS = $(MODS_FOR:.for=.mod)

OFILES = $(SRC_F90:.f90=.o) $(SRC_FOR:.for=.o) 

FOPTS = -fd-lines-as-comments
F90OPTS = -ffree-form -ffree-line-length-none 

# DISPOSABLES: files created during the execution that can safely be deleted
DISPOSABLES = LUN.LST 

%.mod : %.for 
	gfortran $(FOPTS) -c $<

%.o : %.for
	gfortran $(FOPTS) -c $<

%.o : %.f90
	gfortran $(F90OPTS) -c $<

$(DEST) : $(MODS) $(OFILES) 
	gfortran $(FOPTS) $(OFILES) -o $(DEST)


.PHONY: clean
clean:
	/bin/rm -f *.BAK *~ *.mod *.o $(DEST) $(DISPOSABLES)
