# make file to compile and link the Fortran files

DEST = main.out

# SRC_F90: Fortran source files
SRC_F90 = main.f90 hru_module.f90 mgt_operations_module.f90 time_module.f90 \
	hydrograph_module.f90 basin_module.f90 maximum_data_module.f90 \
	conditional_module.f90 climate_module.f90 calibration_data_module.f90 \
	proc_bsn.f90 proc_date_time.f90 proc_db.f90 proc_read.f90 plant_parm_read.f90 \
	plantparm_init.f90 till_parm_read.f90 pest_parm_read.f90 input_file_module.f90 \
	plant_data_module.f90 tillage_data_module.f90 pesticide_data_module.f90 \
	ascrv.f90 soil_lte_db_read.f90 soil_db_read.f90 snowdb_read.f90 hydrol_read.f90 \
	hru_lte_module.f90 soil_data_module.f90 hydrology_data_module.f90 field_read.f90 \
	topo_read.f90 salt_hru_aqu_read.f90 hmet_hru_aqu_read.f90 path_hru_aqu_read.f90 \
	pest_hru_aqu_read.f90 solt_db_read.f90 topography_data_module.f90 \
	constituent_mass_module.f90 soil_plant_init.f90 constit_db_read.f90 \
	cli_staread.f90 cli_read_atmodep.f90 pathogen_data_module.f90 search.f90 \
	landuse_read.f90 overland_n_read.f90 cons_prac_read.f90 cntbl_read.f90 \
	readpcom.f90 scen_read_bmpuser.f90 scen_read_filtstrip.f90 scen_read_grwway.f90 \
	sep_read.f90 septic_data_module.f90 tiles_data_module.f90 landuse_data_module.f90 \
	sdr_read.f90 mgt_read_mgtops.f90 mgt_read_fireops.f90 mgt_read_sweepops.f90 \
	mgt_read_grazeops.f90 mgt_read_harvops.f90 mgt_read_chemapp.f90 mgt_read_irrops.f90 \
	septic_parm_read.f90 path_parm_read.f90 urban_parm_read.f90 fert_parm_read.f90 \
	fertilizer_data_module.f90 urban_data_module.f90 read_mgtops.f90 cli_wgnread.f90 \
	cli_wmeas.f90 cli_hmeas.f90 cli_smeas.f90 cli_tmeas.f90 cli_pmeas.f90 \
	cli_initwgn.f90 gcycl.f90 jdt.f90 aunif.f90 cli_dstn1.f90 basin_print_codes_read.f90 \
	basin_prm_default.f90 basin_read_prm.f90 basin_read_cc.f90 readcio_read.f90 \
	readtime_read.f90 basin_read_objs.f90 organic_mineral_mass_module.f90 \
	calhard_control.f90 cal_parmchg_read.f90 calsoft_control.f90 time_control.f90 \
	command.f90 sd_channel_module.f90 aquifer_module.f90 plant_module.f90 \
	ru_module.f90 output_landscape_module.f90 channel_module.f90 \
	output_ls_pesticide_module.f90 reservoir_module.f90 water_body_module.f90 \
	soil_nutcarb_output.f90 soil_nutcarb_write.f90 basin_recall_output.f90 \
	basin_sdchannel_output.f90 basin_chanmorph_output.f90 basin_channel_output.f90 \
	basin_reservoir_output.f90 basin_aquifer_output.f90 soil_module.f90 lsu_output.f90 \
	basin_output.f90 basin_aqu_pest_output.f90 aqu_pesticide_module.f90 \
	basin_ls_pest_output.f90 basin_res_pest_output.f90 basin_ch_pest_output.f90 \
	hydin_output.f90 recall_output.f90 res_pesticide_module.f90 ch_pesticide_module.f90 \
	ru_output.f90 res_pesticide_output.f90 reservoir_output.f90 cha_pesticide_output.f90 \
	sd_channel_output.f90 sd_chanmorph_output.f90 channel_output.f90 aqu_pesticide_output.f90 \
	aquifer_output.f90 hru_pathogen_output.f90 hru_pesticide_output.f90 wetland_output.f90 \
	hru_output.f90 output_ls_pathogen_module.f90 hru_lte_output.f90 obj_output.f90 \
	hydout_output.f90 flow_dur_curve.f90 sd_channel_control.f90 constit_hyd_mult.f90 \
	res_control.f90 channel_control.f90 aqu_1d_control.f90 hyddep_output.f90 ru_control.f90 \
	hru_lte_control.f90 channel_velocity_module.f90 dr_module.f90 reservoir_data_module.f90 \
	channel_data_module.f90 actions.f90 conditions.f90 pl_burnop.f90 plant_init.f90 \
	pl_graze.f90 pest_apply.f90 mgt_killop.f90 mgt_harvtuber.f90 mgt_harvgrain.f90 \
	mgt_harvbiomass.f90 pl_fert.f90 carbon_module.f90 structure_set_parms.f90 curno.f90 \
	pl_partition.f90 pl_seed_gro.f90 pl_root_gro.f90 xmon.f90 ttcoef_wway.f90 \
	pl_rootfr.f90 mgt_newtillmix.f90 ch_rtout.f90 ch_rchuse.f90 ch_rthsed.f90 \
	ch_watqual3.f90 ch_rtday.f90 ch_rtsed_yangsand.f90 ch_ttcoef.f90 qman.f90 theta.f90 \
	ch_watqual_semi_analitical_function.f90 mgt_tillfactor.f90 ch_rtsed_Molinas_Wu.f90 \
	ch_rtsed_kodatie.f90 ch_rtsed_bagnold.f90 ch_rtsed.f90 ch_rtmusk.f90 ch_rchinit.f90 \
	res_pest.f90 res_nutrient.f90 res_sediment.f90 res_hydro.f90 ch_rtpath.f90 \
	ch_rtpest.f90 ch_watqual4.f90 hru_control.f90 calsoft_ave_output.f90 \
	calsoft_sum_output.f90 cli_atmodep_time_control.f90 hru_hyds.f90 hru_urb_bmp.f90 \
	smp_bmpfixed.f90 smp_grass_wway.f90 smp_buffer.f90 smp_filter.f90 swr_substor.f90 \
	stor_surfstor.f90 swr_latsed.f90 hru_urbanhr.f90 hru_urban.f90 path_ls_process.f90 \
	path_ls_swrouting.f90 regres.f90 hru_sweep.f90 path_ls_runoff.f90 swr_subwq.f90 \
	nut_solp.f90 nut_nlch.f90 nut_nrain.f90 nut_psed.f90 nut_orgnc2.f90 nut_orgnc.f90 \
	nut_orgn.f90 pest_pesty.f90 pest_enrsb.f90 pest_soil_tot.f90 pest_lch.f90 pest_decay.f90 \
	pest_washp.f90 sep_biozone.f90 nut_pminrl2.f90 nut_pminrl.f90 nut_nitvol.f90 \
	cbn_zhang2.f90 nut_nminrl.f90 pl_moisture_senes_init.f90 pl_moisture_gro_init.f90 \
	pl_grow.f90 pl_mortality.f90 pl_leaf_senes.f90 pl_leaf_gro.f90 pl_biomass_gro.f90 \
	pl_dormant.f90 pl_nut_demand.f90 nut_np_flow.f90 nut_denit.f90 fcgd.f90 pl_community.f90 \
	wetland_control.f90 swr_percmain.f90 swr_origtile.f90 swr_drains.f90 swr_satexcess.f90 \
	swr_percmicro.f90 swr_percmacro.f90 pl_waterup.f90 pl_pupd.f90 pl_nupd.f90 pl_pup.f90 \
	pl_nup.f90 pl_tstr.f90 nuts.f90 pl_nfix.f90 swr_depstor.f90 mgt_operatn.f90 \
	et_act.f90 et_pot.f90 surface.f90 stmp_solt.f90 albedo.f90 rls_routetile.f90 \
	rls_routesoil.f90 rls_routesurf.f90 varinit.f90 ero_ysed.f90 ero_cfactor.f90 \
	ero_ovrsed.f90 ero_eiusle.f90 ero_pkq.f90 sq_surfst.f90 sq_crackflow.f90 sq_volq.f90 \
	sq_dailycn.f90 sq_crackvol.f90 sq_snom.f90 sq_canopyint.f90 ee.f90 mgt_sched.f90 \
	climate_control.f90 atri.f90 expo.f90 cli_wndgen.f90 cli_bounds_check.f90 \
	cli_rhgen.f90 cli_slrgen.f90 cli_clgen.f90 cli_tgen.f90 cli_weatgn.f90 \
	cli_precip_control.f90 sq_greenampt.f90 sq_daycn.f90 cli_pgen.f90 cli_pgenhr.f90 \
	sim_initday.f90 sim_inityr.f90 calsoft_chsed.f90 calsoft_sed.f90 calsoft_plant.f90 \
	calsoft_hyd.f90 define_unit_elements.f90 re_initialize.f90 hyd_connect_out.f90 \
	dr_ru.f90 caltsoft_hyd.f90 unit_hyd.f90 proc_open.f90 proc_cal.f90 \
	hru_dtbl_actions_init.f90 proc_cond.f90 proc_allo.f90 proc_cha.f90 wet_initial.f90 \
	proc_res.f90 salt_cha_res_read.f90 path_cha_res_read.f90 pest_cha_res_read.f90 \
	om_water_init.f90 dtbl_flocon_read.f90 dtbl_scen_read.f90 dtbl_res_read.f90 \
	header_snutc.f90 wet_read.f90 wet_read_hyd.f90 res_initial.f90 res_read.f90 \
	res_objects.f90 res_allo.f90 res_read_init.f90 res_read_weir.f90 res_read_nut.f90 \
	res_read_sed.f90 res_read_hyd.f90 rte_read_nut.f90 ls_link.f90 hru_lte_read.f90 \
	hydro_init.f90 proc_hru.f90  plant_all_init.f90 salt_hru_init.f90 \
	pathogen_init.f90 pesticide_init.f90 hru_output_allo.f90 soiltest_all_init.f90 \
	soils_init.f90 topohyd_init.f90 output_ls_salt_module.f90 soil_nutcarb_init.f90 \
	soil_phys_init.f90 soiltest_init.f90 layersplit.f90 sd_channel_surf_link.f90 \
	hrudb_init.f90 hru_read.f90 hru_allo.f90 time_conc_init.f90 allocate_parms.f90 \
	ch_initial.f90 aqu2d_init.f90 channel_allo.f90 sd_hydsed_init.f90 sd_channel_read.f90 \
	ch_read.f90 ch_read_nut.f90 zeroini.f90 zero2.f90 zero1.f90 zero0.f90 ch_read_sed.f90 \
	ch_read_hyd.f90 sd_hydsed_read.f90 ch_read_init.f90 aqu_read_init.f90 aqu_initial.f90 \
	aqu_read.f90 ru_allo.f90 cal_allo_init.f90 ch_read_parms_cal.f90 ch_read_orders_cal.f90 \
	rec_read_elements.f90 res_read_elements.f90 ch_read_elements.f90 aqu_read_elements.f90 \
	pl_read_parms_cal.f90 pl_read_regions_cal.f90 ls_read_parms_cal.f90 lcu_read_softcal.f90 \
	calsoft_read_codes.f90 cal_cond_read.f90 cal_parm_read.f90 header_write.f90 \
	header_path.f90 header_pest.f90 lsu_read_elements.f90 cal_conditions.f90 \
	header_wetland.f90 header_reservoir.f90 header_hyd.f90 header_yield.f90 header_mgt.f90 \
	header_sd_channel.f90 header_aquifer.f90 header_channel.f90 output_landscape_init.f90 \
	dtbl_lum_read.f90 object_read_output.f90 hyd_connect.f90 recall_module.f90 \
	overbank_read.f90 hyd_read_connect.f90 dr_db_read.f90 recall_read.f90 \
	channel_surf_link.f90 aqu2d_read.f90 ru_read.f90 exco_module.f90 dr_read_salt.f90 \
	dr_read_hmet.f90 dr_read_path.f90 dr_read_pest.f90 dr_read_om.f90 ru_read_elements.f90 \
	chg_par.f90 cal_parm_select.f90 exco_db_read.f90 exco_read_salt.f90 exco_read_hmet.f90 \
	exco_read_path.f90 exco_read_pest.f90 exco_read_om.f90 

# MOD_FOR: Fortran source files that contain modules
MOD_FOR = hru_module.f90 mgt_operations_module.f90 time_module.f90 hydrograph_module.f90 \
	basin_module.f90 maximum_data_module.f90 conditional_module.f90 climate_module.f90 \
	calibration_data_module.f90 input_file_module.f90 plant_data_module.f90 \
	tillage_data_module.f90 pesticide_data_module.f90 hru_lte_module.f90 \
	soil_data_module.f90 hydrology_data_module.f90 topography_data_module.f90 \
	constituent_mass_module.f90 pathogen_data_module.f90 septic_data_module.f90 \
	tiles_data_module.f90 landuse_data_module.f90 fertilizer_data_module.f90 \
	urban_data_module.f90 organic_mineral_mass_module.f90 sd_channel_module.f90 \
	aquifer_module.f90 plant_module.f90 ru_module.f90 output_landscape_module.f90 \
	channel_module.f90 output_ls_pesticide_module.f90 reservoir_module.f90 \
	water_body_module.f90 soil_module.f90 aqu_pesticide_module.f90 res_pesticide_module.f90 \
	ch_pesticide_module.f90 output_ls_pathogen_module.f90 channel_velocity_module.f90 \
	dr_module.f90 reservoir_data_module.f90 channel_data_module.f90 carbon_module.f90 \
	output_ls_salt_module.f90 recall_module.f90 exco_module.f90

MODS = $(MOD_FOR:.f90=.mod)

OFILES = $(SRC_F90:.f90=.o)

#FOPTS = -fd-lines-as-comments
F90OPTS = -ffree-form -ffree-line-length-none 

%.mod : %.f90
	gfortran $(F90OPTS) -c $<

%.o : %.f90
	gfortran $(F90OPTS) -c $<

$(DEST) : $(MODS) $(OFILES) 
	gfortran $(F90OPTS) $(OFILES) -o $(DEST)

hydrograph_module.mod : time_module.mod basin_module.mod

.PHONY: clean
clean:
	/bin/rm -f *.BAK *~ *.mod *.o $(DEST)
