# This Makefile has not been fully tested. Please report any errors to
# champlin@email.arizona.edu
#
# =============================================================================
# *Please read all comments before running Makefile*
# WARNING: Running "make build/quantified_CAG.pkl" from scratch (a fresh
# clone of the delphi backcasting branch) took approximately 2-3 hours on my
# machine.
# =============================================================================


all: report.pdf

# =============================================================================
# Downloading and Unzipping Raw Data Directory
# WARNING: Raw.zip is 1.88 GB zipped and 19.04 GB unzipped. Raw directory will
# be deleted after processing.
# =============================================================================

RAW := data/raw
RAW_ZIP := data/raw.zip
DB_EXIST := data/delphi.db

data/raw.zip:
	if [ ! -d $(RAW) ] && [ ! -f $(DB_EXIST) ]; then curl http://vision.cs.arizona.edu/adarsh/data/raw.zip -o $@; fi
	
data/raw: data/raw.zip
	if [ -f $(RAW_ZIP) ]; then unzip $< -d data; rm $<; fi

# =============================================================================
# Begin Data Processing
# =============================================================================


DSSAT_DATA := data/raw/DSSAT/disaggregated_all_20190123_1451
DSSAT_AGGREGATED_DATA := data/raw/DSSAT/SSD_csv

data/delphi.db: create_delphi_db.py\
				data/south_sudan_data.tsv\
				data/un_to_indicators.tsv\
				data/raw/adjectiveData.tsv\
				$(DSSAT_AGGREGATED_DATA)
	python $^ $@
	rm -R data/raw

data:
	mkdir -p $@

data/south_sudan_data_fao.tsv: data/raw/FAOSTAT
	python data_processing/process_FAO_and_WDI_data.py --fao

data/south_sudan_data_wdi.tsv: data/raw/WDI/WDIData.csv
	python data_processing/process_FAO_and_WDI_data.py --wdi

data/south_sudan_data_fewsnet.tsv: data/raw/FEWS
	python data_processing/process_FEWSNET_data.py	

data/south_sudan_data_climis_unicef_ieconomics.tsv: data_processing/process_climis_unicef_ieconomics_data.py
	python $<

dssat: $(DSSAT_DATA) $(DSSAT_AGGREGATED_DATA)

data/south_sudan_data_dssat.tsv: data_processing/process_dssat_data.py $(DSSAT_DATA)
	python $^ $@

# =============================================================================
# Creation of final indicator table and flat list of indicators
# =============================================================================

data/south_sudan_data.tsv: data_processing/process_FAO_and_WDI_data.py\
						   data/south_sudan_data_fao.tsv\
						   data/south_sudan_data_wdi.tsv\
						   data/raw/wm_12_month_evaluation/south_sudan_data_conflict.tsv\
						   data/south_sudan_data_fewsnet.tsv\
						   data/south_sudan_data_climis_unicef_ieconomics.tsv\
						   data/south_sudan_data_dssat.tsv
	python $< --combine

data/indicator_flat_list.txt: data/south_sudan_data.tsv



# =============================================================================
# Creation of concept to indicator mapping
# 
# *Please read before running Makefile*
# Instructions: You must clone/download the master branch from https://github.com/clulab/eidos
# Please also read their instructions for the proper configuration and setup of 
# Eidos. scala, java, and sbt must be downloaded and installed. The most
# current versions of scala and sbt should work, however you cannot use the
# current version of java. openjdk version 1.8.0_212 is known to work with
# this build. You will need to reconfigure the below pathways in accordance to
# where you downloaded/cloned EIDOS. You must run "sbt assembly" in "/eidos"
# before running this portion of the Makefile. 
#
# WARNING: EIDOS download and setup requires initially approximately 26.94 GB,
# however their instructions indicates options for optimizing and saving space.
# =============================================================================

EIDOS := /Volumes/lorens_drive/eidos
EIDOS_SRC := $(EIDOS)/src/main
ONTOLOGY_GEN_SCRIPT := $(EIDOS_SRC)/python/mk_yaml_ontology.py
TARGET_ONTOLOGY_FILENAME := $(EIDOS_SRC)/resources/org/clulab/wm/eidos/english/ontologies/mitre12_indicators.yml
TARGET_ONTOLOGY_NAME := MITRE12

data/un_to_indicators.tsv: data/indicator_flat_list.txt $(ONTOLOGY_GEN_SCRIPT) data/south_sudan_data.tsv
	python $(ONTOLOGY_GEN_SCRIPT) $< $(TARGET_ONTOLOGY_FILENAME) $(TARGET_ONTOLOGY_NAME) 
	cd $(EIDOS); time java -Xmx50g -cp target/scala-2.12/eidos-assembly-0.2.3-SNAPSHOT.jar org.clulab.wm.eidos.apps.CacheOntologies
	cp $(EIDOS_SRC)/resources/org/clulab/wm/eidos/english/ontologies/$(@F) $@


# =============================================================================
# Creation of reference evaluation CAGs
# =============================================================================

# Via scripting ---------------------------------------------------------------

build:
	mkdir -p $@

PREASSEMBLED_CORPUS := data/raw/wm_12_month_evaluation/wm_12_month_4_reader_20190118.json

build/pruned_corpus.json: data/raw evaluations/create_pruned_corpus.py $(PREASSEMBLED_CORPUS) | build 
	python evaluations/create_pruned_corpus.py $(PREASSEMBLED_CORPUS) $@

build/all_sts.pkl: evaluations/create_all_sts.py build/pruned_corpus.json
	python $^ $@

build/reference_cag.pkl: evaluations/create_reference_cag.py build/all_sts.pkl
	python $^ $@

build/precipitation_centered_CAG.pkl: evaluations/create_precipitation_centered_CAG.py\
									  build/reference_cag.pkl
	python $^ $@

build/CAG_with_indicators.pkl: evaluations/create_CAG_with_indicators.py\
							   build/precipitation_centered_CAG.pkl\
							   data/delphi.db
	python $< build/precipitation_centered_CAG.pkl $@

build/parameterized_CAG.pkl: evaluations/create_parameterized_CAG.py\
							 build/CAG_with_indicators.pkl\
							 data/delphi.db
	python $^ $@

build/quantified_CAG.pkl: evaluations/create_quantified_CAG.py build/parameterized_CAG.pkl
	python $^ $@

report.pdf: report.tex apnotes.cls build/quantified_CAG.pkl
	latexmk -lualatex $<
	latexmk -c $(*F)

insert_model: evaluations/insert_model_into_db.py build/quantified_CAG.pkl data/delphi.db
	python $^

build/delphi_cag.json: evaluations/export_to_sri_and_cra.py build/quantified_CAG.pkl
	python $^ $@
	#
# Via manual construction in the CauseMos HMI ---------------------------------

data/CauseMosCAG.json: | data
	curl http://vision.cs.arizona.edu/adarsh/$(@F).gz | gunzip > $@

build/causemos_cag.pkl: evaluations/create_cag_from_causemos_output.py\
						data/CauseMosCAG.json | data/delphi.db
	mkdir -p build
	python $^ $@

# For Uncharted folks - invoke this as 
#
# make -o data/delphi.db insert_causemos_model
#
# after you have put the newest CauseMos CAG in data/CauseMosCAG.json
insert_causemos_model: evaluations/insert_model_into_db.py build/causemos_cag.pkl | data/delphi.db
	python $^
