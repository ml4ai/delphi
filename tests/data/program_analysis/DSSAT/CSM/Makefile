# make file to compile and link the Fortran files

DEST = csm.out
SRC_FOR = CSMVersion.for OSDefinitions.for ModuleDefs.for OPHEAD.for \
	CSM.for DATES.for UTILS.for READS.for ERROR.for Warning.for \
	PATH.for input_sub.for ipexp.for IPSLIN.for IPSIM.for IPSOIL_Inp.for \
	IPENV.for IPCHEM.for IPTILL.for LMATCH.for Info.for RunList.for \
	IPVAR.for IPMAN.for INTRO.for INSOIL.for SENS.for WEATHR_Inp.for \
	optempy2k.for OPTEMPXY2K.for OPGEN.for SETIME.for SEVAR.for SEWTH.for \
	SECLI.for SECROP.for SERES.for IPECO.for SESIM.for SEFLD.for \
	SEFREQ.for SEINIT.for SEPLT.for SESOIL.for SEHARV.for SEIRR.for \
	SEFERT.for SEPEST.for INVAR.for WTHSET.for CSUTS.for INSTGE.for LAND.for \
	OPSUM.for AUTHAR.for MgmtOps.for AUTPLT.for Fert_Place.for CHEMICAL.for \
	IPSOIL.for OM_Place.for Tillage.for IRRIG.for Paddy_Mgmt.for \
	Flood_Irrig.for OpFlood.for IPIBS.for weathr.for SOIL.for IPWTH_alt.for \
	WTHMOD.for WGEN.for SOLAR.for SPAM.for plant.for SOILDYN.for CO2VAL.for \
	HMET.for OPWEATH.for OPSTRESS.for WATBAL.for SoilMixing.for CENTURY.for \
	N2O_mod.for SOILNI.for SoilPi.for SoilKi.for SoilOrg.for ETPHOT.for \
	STEMP_EPIC.for STEMP.for ROOTWU.for SOILEV.for TRANS.for MULCHEVAP.for \
	SPSUBS.for PET.for ESR_SoilEvap.for CROPGRO.for forage.for CSCERES_Interface.for \
	CSCRP_Interface.for CSCAS_Interface.for CSYCA_Interface.for WH_APSIM.for \
	ML_CERES.for MZ_CERES.for BS_CERES.for PT_SUBSTOR.for RICE.for SC_CNGRO.for \
	CSP_CASUPRO.for SG_CERES.for TR_SUBSTOR.for Aloha_PINE.for SC_CNG_mods.for \
	RETC_VG.for SC_COEFFS.for Aloha_OPHARV.for Aloha_GROSUB.for Aloha_ROOTGR.for \
	Aloha_PHENOL.for SLigCeres.for Aloha_NUPTAK.for Aloha_NFACTO.for OPVIEW.for \
	HResCeres.for TR_OPGROW.for TR_Opharv.for TR_Grosub.for TR_Rootgr.for \
	TR_Nuptak.for TR_Tillsub.for TR_Calcshk.for TR_Nfacto.for TR_Transpl_g.for \
	TR_Ipcrop.for TR_Phenol.for RI_GNURSE.for TR_Transpl_p.for SG_OPHARV.for \
	SG_GROSUB.for SG_ROOTGR.for SG_PHENOL.for SG_PHASEI.for P_CERES.for P_Plant.for \
	RootSoilVol.for OpPlantP.for PPlantSubs.for P_Uptake.for P_IPPLNT.for \
	SG_NUPTAK.for SG_NFACT.for MZ_OPNIT.for MZ_OPGROW.for PEST.for OPPEST.for \
	ROOTDM.for VEGDM.for SEEDDM.for ASMDM.for PESTCP.for LINDM.for IPPROG.for \
	IPPARM.for IPPEST.for CSP_HRes.for CSP_OPHARV.for P_CASUPRO.for CSP_OPGROW.for \
	CSP_SENES.for CSP_ROOTS.for CSP_GROW_CANE.for CSP_NUPTAK.for CSP_RESPIR.for \
	CSP_PHOTO.for CSP_PHENOL.for CSP_INCOMP.for PlantNBal.for CSP_INCOMP_OUT.for \
	CSP_INPHENOL.for CSP_IPPHENOL.for CSP_CANOPY.for CSP_IPPLNT.for SC_CCOUT.for \
	SC_OPHARV.for SC_PHOTOS.for SC_OUTPUT.for SC_ETOUT.for SC_PHENOL.for SC_PARTIT.for \
	SC_Poplt3.for CANOPY.for SC_GTP_SHOOTS.for SC_ROOTG.for SC_Canop3.for \
	RI_Grosub.for RI_Opharv.for RI_Phenol.for RI_Rootgr.for RI_Transpl_p.for \
	RI_Tillsub.for RI_Nfacto.for RI_Nuptak.for MZ_KUPTAK.for RI_KUPTAK.for \
	RI_Ipcrop.for RI_Calcshk.for RI_Transpl_g.for RI_Opgrow.for PT_OPGROW.for \
	PT_OPHARV.for PT_GROSUB.for PT_ROOTGR.for PT_PHENOL.for PT_PHASEI.for \
	PT_THTIME.for PT_NUPTAK.for PT_NFACTO.for BS_OPHARV.for BS_OPNIT.for \
	BS_OPGROW.for BS_GROSUB.for BS_ROOTS.for BS_PHENOL.for BS_NUPTAK.for \
	BS_NFACTO.for MZ_OPHARV.for MZ_GROSUB.for MZ_IX_GROSUB.for MZ_ROOTS.for \
	MZ_PHENOL.for SW_GROSUB.for SW_FreshWt.for MZ_NUPTAK.for MZ_NFACTO.for \
	MZ_IX_NUPTAK.for MZ_IX_LEAFAREA.for MZ_IX_KNUMBER.for MZ_IX_PHOTSYNT.for \
	MZ_IX_PHENOL.for MZ_IX_RESPIR.for MZ_IX_PLANTG.for MZ_IX_RADABS.for ML_OPGROW.for \
	ML_opharv.for ML_rootgr.for ML_GROSUB.for ML_PHENOL.for ML_PHASEI.for \
	ML_NUPTAK.for ML_TILLSUB.for ML_NFACT.for WH_OPHARV.for WH_OPGROW.for \
	WH_GROSUB.for WH_PHENOL.for WH_OPNIT.for WH_COLD.for WH_temp.for WH_SW_SUBS.for \
	WH_NUPTAK.for WBSUBS.for WH_NFACTO.for WH_ROOTS.for CSCAS.for CSCRP.for CSCER.for \
	CSREADS.for CSDISEASE.for for_hres_cgro.for for_plantnbal.for for_dormancy.for \
	for_opgrow.for for_opmob.for for_pest.for for_oppest.for for_rootdm.for for_vegdm.for \
	for_seeddm.for for_pestcp.for for_lindm.for for_asmdm.for for_ipprog.for \
	for_ippest.for for_ipparm.for for_harv.for for_grow.for for_roots.for for_freeze.for \
	for_veggr.for for_poddet.for for_canopy.for for_pods.for for_nfix.for for_mobil.for \
	for_nuptak.for for_demand.for for_sdcomp.for for_ch2oref.for for_respir.for \
	for_senmob.for for_incomp.for for_phenol.for for_photo.for for_ipplnt.for \
	for_rstages.for for_ipphenol.for HRes_CGRO.for OPHARV.for P_CGRO.for Opgrow.for \
	PODS.for FreshWt.for GROW.for ROOTS.for FREEZE.for SENES.for VEGGR.for PODDET.for \
	NFIX.for MOBIL.for NUPTAK.for DEMAND.for PHENOL.for PHOTO.for INCOMP.for IPPLNT.for \
	RStages.for SDCOMP.for RESPIR.for OpStemp.for Ipphenol.for OPETPHOT.for \
	ETPHR.for SoilNoPoBal.for OpSoilOrg.for SoilNBalSum.for MULCHLAYER.for NCHECK_organic.for \
	SoilOrg_init.for OpSoilKi.for SoilK_init.for OpSoilPi.for SoilPiBal.for SoilPi_init.for \
	SoilNiBal.for SoilPBalSum.for IPHedley_inorg.for Flood_Chem.for OPSOILNI.for \
	NCHECK_inorg.for OXLAYER.for NFLUX.for EQUIL2.for FCHEM.for FLOODI.for OPFLOODN.for \
	Denit_DayCent.for Denit_Ceres.for diffusiv.for SoilNi_init.for SoilPoBal.for \
	SoilPoBal_C.for OPSOMLIT_C.for SOMLITPRINT_C.for SoilNoBal.for SoilNoBal_C.for \
	SoilCBal_C.for NCHECK_C.for TSOMLIT_C.for IMMOBLIMIT_C.for SOMDEC_C.for LITDEC_C.for \
	CE_RATIO_C.for DECRAT_C.for INCORPOR_C.for SENESADD_C.for RPLACE_C.for PARTIT_C.for \
	EFLOW_C.for SoilCNPinit_C.for SOMINIT_c.for SOMFIX_C.for IPHedley_C.for MULCHWAT.for \
	WBAL.for OPMULCH.for OPWBAL.for TILEDRAIN.for SATFLO.for INFIL.for RNOFF.for \
	TillEvent.for TextureClass.for OR_Opharv.for rddtmp.for addstr.for addint.for istart.for \
	rddata.for ambusy.for flexist.for extens.for messwrt.for flname.for swpi4.for rdtmp2.for \
	warning_OR.for lowerc.for entdch.for rdindx.for rdtmp1.for rderr.for rdsctb.for \
	rdlex.for dtsys.for dtleap.for fopens.for rderri.for enthlp.for addinf.for entcha.for \
	str_copy.for getun.for addrea.for addref.for outcom.for addstf.for LINT2.for notnul.for \
	INTGRL.for rdsrea.for rdinqr.for decrec.for getrec.for words.for rdsint.for rdarea.for \
	rdinlv.for rdinit.for decrea.for INSW.for parsword.for rdscha.for LIMIT.for

SRC_F90 = csvlinklist.f90 csvoutput.f90 YCA_Albedo_Check_m.f90 YCA_First_Trans_m.f90 \
	YCA_Node.f90 WH_module.f90 ORYZA_Interface.f90 VARIABLE.f90 OR_ROOTG.f90 \
	Aloha_mod.f90 Aloha_OPGROW.f90 CSYCA.f90 nox_pulse.f90 YCA_Output.f90 \
	YCA_Formats_m.f90 YCA_Out_ReInit.f90 YCA_Out_StoreVars.f90 YCA_Out_Sens.f90 \
	YCA_Out_CrpSim.f90 YCA_Out_Error.f90 YCA_Out_WrkPhenRes.f90 YCA_Out_LfTier.f90 \
	YCA_Out_PlantSum.f90 YCA_Out_Eval.f90 YCA_Out_PlGrow.f90 YCA_Control_Plant.f90 \
	YCA_Out_Work.f90 YCA_Out_ModFail.f90 YCA_Integrate.f90 YCA_Growth.f90 \
	YCA_SeasInit.f90 YCA_RunInit.f90 YCA_Control_Leaf.f90 YCA_Control_Environment.f90 \
	YCA_SeasInit_Final.f90 YCA_SeasInit_SetStage.f90 YCA_SeasInit_ReadGeno.f90 \
	YCA_SeasInit_PlHarvDat.f90 YCA_SeasInit_ReadXfile.f90 YCA_SeasInit_VarInit.f90 \
	YCA_Growth_Distribute.f90 YCA_Growth_NUptake.f90 YCA_Growth_Part.f90 \
	YCA_Growth_Photo.f90 YCA_Growth_Senesce.f90 YCA_Growth_Rates.f90 YCA_Growth_Init.f90 \
	YCA_Growth_Evapo.f90 YCA_PrePlant.f90 YCA_Control_Photosynthesis.f90 \
	YCA_Control_VPDEffect.f90 YCA_Integ_EndCrop.f90 YCA_Integ_WthrSum.f90 \
	YCA_Integ_SeasEnd.f90 YCA_Integ_HstFail.f90 YCA_Integ_Nconc.f90 YCA_Integ_Stages.f90 \
	YCA_Integ_N.f90 YCA_Integ_LA.f90 YCA_Integ_AgesWts.f90 OR_Opgrow.f90 Ncrop3.f90 \
	outdat.f90 fatalerr.f90 ttutilprefs.f90 fopengstandard.f90 ifindc.f90 upperc.f90 \
	messinq.f90 recread.f90 recreadt.f90 recreadi.f90 OPSTOR.f90 INTGR2.f90 OBSSYS.f90 \
	OPSYS.f90 TTHEAD.f90 DS1900.f90 Nnostress2.f90 ORYZA1.f90 wnostress.f90 GETOBS.f90 \
	INQOBS.f90 SUBCBC.f90 SUBLAI3.f90 SUBDD.f90 PARTITION.f90 NightT.f90 SUBGRN.f90 \
	SGPCDT.f90 GPPARSET.f90 SGPC2.f90 SGPC1.f90 SSKYC.f90 SASTRO.f90 SGPL.f90 GPPARGET.f90 \
	SRDPRF.f90 SUBCD2.f90 OR_PHENOL.f90 WStress2.f90 G_Soil.f90 G_Exper.f90 SVPS1.f90 \
	quadpack.f90 OBSINI.f90

MODS = csmversion.mod osdefinitions.mod moduledefs.mod summodule.mod \
	headermod.mod csvoutput.mod linklist.mod interface_ipsoil.mod \
	modsoilmix.mod n2o_mod.mod yca_albedo_check_m.mod yca_first_trans_m.mod \
	yca_node.mod wh_module.mod public_module.mod rootgrowth.mod \
	cng_moduledefs.mod aloha_mod.mod interface_senlig_ceres.mod interface_soilnbalsum.mod \
	interface_soilpbalsum.mod yca_formats_m.mod yca_control_plant.mod yca_control_leaf.mod \
	yca_control_environment.mod yca_control_photosyntesis.mod yca_control_vpdeffect.mod \
	module_outdat.mod ttutilprefs.mod


OFILES = $(SRC_F90:.f90=.o) $(SRC_FOR:.for=.o) 

FOPTS = -fd-lines-as-comments
F90OPTS = -ffree-form -ffree-line-length-none 

%.mod : %.for 
	gfortran $(FOPTS) -c $<

%.o : %.for
	gfortran $(FOPTS) -c $<

%.o : %.f90
	gfortran $(F90OPTS) -c $<

$(DEST) : $(MODS) $(OFILES) 
	gfortran $(FOPTS) $(OFILES) -o $(DEST)

moduledefs.mod : csmversion.mod osdefinitions.mod ModuleDefs.for
	gfortran -c ModuleDefs.for

csmversion.mod : CSMVersion.for
	gfortran -c CSMVersion.for 

osdefinitions.mod : OSDefinitions.for
	gfortran -c OSDefinitions.for

headermod.mod : moduledefs.mod moduledata.mod OPHEAD.for

moduledata.mod floodmodule.mod : moduledefs.mod ModuleDefs.for

interface_ipsoil.mod : IPSOIL.for
	gfortran -c IPSOIL.for

cng_moduledefs.mod SC_CNG_mods.o : SC_CNG_mods.for
	gfortran -c SC_CNG_mods.for

summodule.mod : OPSUM.for

modsoilmix.mod : SoilMixing.for

n2o_mod.mod : N2O_mod.for
	gfortran -c N2O_mod.for

wh_module.mod : WH_module.f90

csvoutput.mod csvoutput.o : csvoutput.f90 linklist.mod
	gfortran -c csvoutput.f90

linklist.mod csvlinklist.o : csvlinklist.f90
	gfortran -c csvlinklist.f90

yca_albedo_check_m.mod YCA_Albedo_Check_m.o : YCA_Albedo_Check_m.f90
	gfortran -c YCA_Albedo_Check_m.f90

yca_first_trans_m.mod YCA_First_Trans_m.o : yca_node.mod YCA_First_Trans_m.f90
	gfortran -c YCA_First_Trans_m.f90

yca_node.mod YCA_Node.o : YCA_Node.f90
	gfortran -c YCA_Node.f90

WH_module.o wh_module.mod : WH_module.f90
	gfortran -c WH_module.f90

ORYZA_Interface.o : moduledefs.mod floodmodule.mod moduledata.mod public_module.mod \
	rootgrowth.mod interface_ipsoil.mod ORYZA_Interface.f90
	gfortran -c ORYZA_Interface.f90

public_module.mod VARIABLE.o : VARIABLE.f90
	gfortran -c VARIABLE.f90

rootgrowth.mod OR_ROOTG.o : OR_ROOTG.f90
	gfortran -c OR_ROOTG.f90

aloha_mod.mod : Aloha_mod.f90
	gfortran -c Aloha_mod.f90

interface_senlig_ceres.mod : SLigCeres.for
	gfortran -c SLigCeres.for

interface_soilnbalsum.mod: SoilNBalSum.for
	gfortran -c SoilNBalSum.for

interface_soilpbalsum.mod : SoilPBalSum.for
	gfortran -c SoilPBalSum.for

yca_formats_m.mod : YCA_Formats_m.f90
	gfortran -c YCA_Formats_m.f90

yca_control_plant.mod : YCA_Control_Plant.f90
	gfortran -c YCA_Control_Plant.f90

yca_control_leaf.mod : YCA_Control_Leaf.f90
	gfortran $(F90OPTS) -c YCA_Control_Leaf.f90

yca_control_environment.mod : YCA_Control_Environment.f90
	gfortran $(F90OPTS) -c YCA_Control_Environment.f90

yca_control_photosyntesis.mod YCA_Control_Photosynthesis.o : yca_control_vpdeffect.mod YCA_Control_Photosynthesis.f90
	gfortran $(F90OPTS) -c YCA_Control_Photosynthesis.f90

yca_control_vpdeffect.mod : YCA_Control_VPDEffect.f90
	gfortran $(F90OPTS) -c YCA_Control_VPDEffect.f90

module_outdat.mod : outdat.f90
	gfortran $(F90OPTS) -c outdat.f90

ttutilprefs.mod : ttutilprefs.f90
	gfortran $(F90OPTS) -c ttutilprefs.f90

WH_APSIM.o : wh_module.mod WH_APSIM.for

CSM.o : headermod.mod CSM.for
MgmtOps.o : moduledefs.mod floodmodule.mod summodule.mod MgmtOps.for
CSYCA_Interface.o : yca_first_trans_m.mod CSYCA_Interface.for

.PHONY: clean
clean:
	/bin/rm -f *.BAK *~ *.mod *.o $(DEST)
