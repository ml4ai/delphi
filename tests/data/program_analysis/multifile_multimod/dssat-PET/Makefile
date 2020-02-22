# make file to compile and link the Fortran files

DEST = csm.out
SRC_FOR = CSMVersion.for OSDefinitions.for ModuleDefs.for CSM.for LAND.for \
	SOIL.for WATBAL.for MULCHWAT.for OPMULCH.for DATES.for WBAL.for \
	UTILS.for OPWBAL.for OPSUM.for READS.for WBSUBS.for TILEDRAIN.for \
	SATFLO.for INFIL.for RNOFF.for SOILNI.for SoilKi.for SoilOrg.for \
	N2O_mod.for IPSOIL.for PATH.for SoilNoPoBal.for SoilNBalSum.for \
	OpSoilOrg.for MULCHLAYER.for NCHECK_organic.for SoilOrg_init.for \
	OpSoilKi.for SoilK_init.for SoilNiBal.for Flood_Chem.for OPSOILNI.for \
	NCHECK_inorg.for OXLAYER.for NFLUX.for Denit_Ceres.for diffusiv.for \
	SoilNi_init.for SoilPi.for Info.for OPHEAD.for SOILDYN.for SoilMixing.for \
	TillEvent.for LMATCH.for RETC_VG.for TextureClass.for MgmtOps.for Paddy_Mgmt.for \
	OpFlood.for OM_Place.for AUTHAR.for IRRIG.for Flood_Irrig.for AUTPLT.for \
	CHEMICAL.for Tillage.for Fert_Place.for plant.for CROPGRO.for HRes_CGRO.for \
	PlantNBal.for OPHARV.for Opgrow.for PODS.for GROW.for ROOTS.for SENES.for \
	VEGGR.for NFIX.for ERROR.for Warning.for CANOPY.for FreshWt.for OPVIEW.for \
	OPSTRESS.for MOBIL.for NUPTAK.for DEMAND.for PEST.for OPPEST.for ROOTDM.for \
	VEGDM.for SEEDDM.for ASMDM.for PESTCP.for LINDM.for IPPROG.for IPPARM.for \
	IPPEST.for SDCOMP.for RESPIR.for P_CGRO.for PHENOL.for PHOTO.for RStages.for \
	Ipphenol.for RootSoilVol.for P_Plant.for OpPlantP.for PPlantSubs.for P_Uptake.for \
	P_IPPLNT.for INCOMP.for IPPLNT.for SPAM.for ETPHOT.for STEMP.for SPSUBS.for \
	OpStemp.for OPETPHOT.for ETPHR.for TRANS.for HMET.for ESR_SoilEvap.for \
	MULCHEVAP.for PET.for ROOTWU.for weathr.for IPWTH_alt.for WTHMOD.for \
	OPWEATH.for SOLAR.for CO2VAL.for IPIBS.for RunList.for input_sub.for OPGEN.for \
	OPTEMPXY2K.for WEATHR_Inp.for WTHSET.for CSUTS.for optempy2k.for INSOIL.for \
	INTRO.for ipexp.for IPSOIL_Inp.for IPVAR.for IPSLIN.for IPMAN.for IPSIM.for \
	IPENV.for IPCHEM.for IPTILL.for 

SRC_F90 = nox_pulse.f90

MODS_FOR = CSMVersion.for OSDefinitions.for ModuleDefs.for N2O_mod.for IPSOIL.for \
	SoilNBalSum.for OPHEAD.for 

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
