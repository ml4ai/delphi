SRCFILES = $(wildcard *.f)
EXEFILES = $(SRCFILES:.f=.exe)
OUTFILES = $(SRCFILES:.f=.OUT)

%.exe:%.f
	gfortran $*.f -o $*.exe

%.OUT:%.exe
	$*.exe > $@

.PHONY: exe, out, clean

# exe : create executable files
exe : $(EXEFILES)

# out : run the executables and create output files
out : $(OUTFILES)

# clean: remove executables and clutter
clean:
	/bin/rm -f *.BAK $(EXEFILES)

# realclean: remove output files as well
realclean:
	/bin/rm -f *.BAK $(EXEFILES) $(OUTFILES)
