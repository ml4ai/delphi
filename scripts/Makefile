all: data/delphi.db

data/delphi.db: create_delphi_db.py\
				data/south_sudan_data.tsv\
				data/un_to_indicators.tsv
	python $^ $@
	#cp $@ $(DELPHI_DATA)/$(@F)

data:
	mkdir -p $@

# Intermediate files that will be deleted after they are no longer necessary

.INTERMEDIATE: data/WDI_csv.tar.bz2 \
			   data/FAOSTAT.zip\
			   data/FEWSNET/ALL_HFIC.zip\
			   data/FEWSNET/FEWSNET_World_Admin.zip\
			   fews


# =============================================================================
# WDI data processing
# =============================================================================

data/WDI_csv.tar.bz2: | data
	curl http://vision.cs.arizona.edu/adarsh/$(@F) -o $@

data/WDI_csv/WDIData.csv: data/WDI_csv.tar.bz2
	tar -C $(<D) -jxvf $<


# =============================================================================
# FAOSTAT data processing
# =============================================================================

data/FAOSTAT.zip: | data
	curl http://fenixservices.fao.org/faostat/static/bulkdownloads/$(@F) -o $@

data/FAOSTAT: data/FAOSTAT.zip
	unzip $< -d $@
	cd $@; ls *.zip | parallel --bar -k unzip 

# =============================================================================
# FEWSNET data processing
# =============================================================================

data/FEWSNET/FEWSNET_World_Admin.zip: | data
	mkdir -p $(@D)
	curl http://shapefiles.fews.net.s3.amazonaws.com/ADMIN/$(@F) -o $@

data/FEWSNET/FEWSNET_World_Admin: data/FEWSNET/FEWSNET_World_Admin.zip
	unzip $< -d $@

data/FEWSNET/ALL_HFIC.zip:
	curl http://shapefiles.fews.net.s3.amazonaws.com/$(@F) -o $@

data/FEWSNET/ALL_HFIC: data/FEWSNET/ALL_HFIC.zip
	unzip $< -d $(@D)
	cd $@/East\ Africa; for f in EA2017*; do mv $$f $${f/EA/EA_}; done

fews: data/FEWSNET/FEWSNET_World_Admin data/FEWSNET/ALL_HFIC

# =============================================================================
# Intermediate indicator file creation
# =============================================================================

data/south_sudan_data_fao.tsv: data/FAOSTAT
	python process_FAO_and_WDI_data.py --fao

data/south_sudan_data_wdi.tsv: data/WDI_csv/WDIData.csv
	python process_FAO_and_WDI_data.py --wdi

data/south_sudan_data_fewsnet.tsv: process_FEWSNET_data.py fews
	python $<

data/south_sudan_data_climis_unicef_ieconomics.tsv: process_climis_unicef_ieconomics_data.py
	python $<

data/south_sudan_data_conflict.tsv:
	curl http://vision.cs.arizona.edu/adarsh/$(@F) -o $@

DSSAT_DATA := ~/Downloads/disaggregated_all_20190123_1451
data/south_sudan_data_dssat.tsv: process_dssat_data.py $(DSSAT_DATA)
	python $^ $@

# =============================================================================
# Creation of final indicator table and flat list of indicators
# =============================================================================

data/south_sudan_data.tsv: process_FAO_and_WDI_data.py\
						   data/south_sudan_data_fao.tsv\
						   data/south_sudan_data_wdi.tsv\
						   data/south_sudan_data_conflict.tsv\
						   data/south_sudan_data_fewsnet.tsv\
						   data/south_sudan_data_climis_unicef_ieconomics.tsv\
						   data/south_sudan_data_dssat.tsv

	python $< --combine

data/indicator_flat_list.txt: data/south_sudan_data.tsv

# =============================================================================
# Creation of concept to indicator mapping
# =============================================================================

EIDOS := ~/clulab/eidos
EIDOS_SRC := $(EIDOS)/src/main
ONTOLOGY_GEN_SCRIPT := $(EIDOS_SRC)/python/mk_yaml_ontology.py
TARGET_ONTOLOGY_FILENAME := $(EIDOS_SRC)/resources/org/clulab/wm/eidos/english/ontologies/mitre12_indicators.yml
TARGET_ONTOLOGY_NAME := MITRE12

data/un_to_indicators.tsv: data/indicator_flat_list.txt $(ONTOLOGY_GEN_SCRIPT) data/south_sudan_data.tsv
	python $(ONTOLOGY_GEN_SCRIPT) $< $(TARGET_ONTOLOGY_FILENAME) $(TARGET_ONTOLOGY_NAME) 
	cd $(EIDOS); time sbt "runMain org.clulab.wm.eidos.apps.CacheOntologies"
	cp $(EIDOS_SRC)/resources/org/clulab/wm/eidos/english/ontologies/$(@F) $@


# =============================================================================
# Creation of reference evaluation CAG
# =============================================================================

build:
	mkdir -p $@

PREASSEMBLED_CORPUS := $(HOME)/Downloads/wm_12_month_4_reader_20190118.json

build/pruned_corpus.json: evaluations/create_pruned_corpus.py $(PREASSEMBLED_CORPUS) | build
	python $^ $@

build/all_sts.pkl: evaluations/create_all_sts.py build/pruned_corpus.json
	python $^ $@

build/reference_cag.pkl: evaluations/create_reference_cag.py build/all_sts.pkl
	python $^ $@

build/precipitation_centered_CAG.pkl: evaluations/create_precipitation_centered_CAG.py\
									  build/reference_cag.pkl
	python $^ $@

build/CAG_with_indicators.pkl: evaluations/create_CAG_with_indicators.py\
							   build/precipitation_centered_CAG.pkl
	python $^ $@

build/parameterized_CAG.pkl: evaluations/create_parameterized_CAG.py\
							 build/CAG_with_indicators.pkl\
							 #data/delphi.db
	python $^ $@

report.pdf: report.tex apnotes.cls build/parameterized_CAG.pkl
	latexmk -lualatex $<
	latexmk -c $(*F)
