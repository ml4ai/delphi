# This Makefile has not been fully tested. Please report any errors to
# champlin@email.arizona.edu
# Last edited: 6/20/2019
#
# =============================================================================
# *Please read all comments before running Makefile*
# WARNING: Running "make build/quantified_CAG.pkl" from scratch (a fresh
# clone of the delphi backcasting branch) took approximately 2-3 hours on my
# machine.
#
# You will also need to set the DELPHI_DATA enviroment variable to point to the
# scripts/data directory.
# =============================================================================


all: report.pdf

# =============================================================================
# Downloading and Unzipping Raw Data Directory
# WARNING: Raw.zip is 1.88 GB zipped and 19.04 GB unzipped. Raw directory will
# be deleted after processing.
# =============================================================================

RAW := data/raw
RAW_ZIP := $(RAW).zip


$(RAW_ZIP):
	mkdir -p data
	curl http://vanga.sista.arizona.edu/delphi_data/$(@F) -o $@
	
$(RAW): $(RAW_ZIP)
	unzip $< -d data

# =============================================================================
# Begin Data Processing
# =============================================================================


DSSAT_DATA := $(RAW)/DSSAT/disaggregated_all_20190123_1451
DSSAT_AGGREGATED_DATA := $(RAW)/DSSAT/SSD_csv

data/delphi.db: create_delphi_db.py\
				data/indicator_data.tsv\
				data/primary_to_indicators.tsv\
				data/raw/adjectiveData.tsv\
				$(DSSAT_AGGREGATED_DATA)
	python $^ $@

data:
	mkdir -p $@

$(RAW)/FAOSTAT: $(RAW)
$(RAW)/WDI/WDIData.csv: $(RAW)
$(RAW)/FEWS: $(RAW)
PREASSEMBLED_CORPUS := $(RAW)/wm_12_month_evaluation/jsonld-merged20190404.json
$(PREASSEMBLED_CORPUS): $(RAW)
$(DSSAT_DATA): $(RAW)
$(RAW)/wm_12_month_evaluation/indicator_data_conflict.tsv: $(RAW)
$(RAW)/adjectiveData.tsv: $(RAW)
$(DSSAT_AGGREGATED_DATA): $(RAW)


data/fao_data.tsv: data_processing/process_FAO_data.py $(RAW)/FAOSTAT
	python $< $@

data/wdi_data.tsv: data_processing/process_WDI_data.py $(RAW)/WDI/WDIData.csv
	python $^ $@

data/indicator_data_fewsnet.tsv: data_processing/process_FEWSNET_data.py $(RAW)/FEWS 
	python $< $@

data/indicator_data_climis_unicef_ieconomics.tsv: data_processing/process_climis_unicef_ieconomics_data.py
	python $< $@

dssat: $(DSSAT_DATA) $(DSSAT_AGGREGATED_DATA)

data/indicator_data_dssat.tsv: data_processing/process_dssat_data.py $(DSSAT_DATA)
	python $^ $@
	
data/indicator_data_UNHCR.tsv: data_processing/process_UNHCR_asylum_seeker_data.py
	python $< $@

data/south_sudan_migration_data_initial.tsv: data_processing/process_migration_data_initial.py
	python $< $@

data/south_sudan_migration_data_secondary.tsv: data_processing/process_migration_data_secondary.py\
										   data/south_sudan_migration_data_initial.tsv
	python $^ $@


data/WHO-data1.csv: data_processing/WHO-csv.py $(RAW)/data_for_november_2019_evaluation/south_sudan_data/WHO
	python $< $@ 

data/WHO-data2.csv: data_processing/WHO-excel.py $(RAW)/data_for_november_2019_evaluation/south_sudan_data/WHO
	python $< 

data/WHO-data3.csv: data_processing/WHO-excel1.py $(RAW)/data_for_november_2019_evaluation/south_sudan_data/WHO
	python $< 

data/IMF-data.csv: data_processing/IMF.py $(RAW)/data_for_november_2019_evaluation/south_sudan_data/IMF
	python $< 

data/WFP-data.csv: data_processing/WFP.py $(RAW)/data_for_november_2019_evaluation/south_sudan_data/WFP
	python $< 

data/World-Bank-data.csv: data_processing/World-Bank.py $(RAW)/data_for_november_2019_evaluation/south_sudan_data/World\ Bank
	python $< 

data/acled-data1.csv: data_processing/acled-South-Sudan.py $(RAW)/data_for_november_2019_evaluation/south_sudan_data/Humanitarian\ Data\ Exchange
	python $< 


data/acled-data2.csv: data_processing/acled-Ethiopia.py $(RAW)/data_for_november_2019_evaluation/ethiopia_east_africa_data/Armed\ Conflict\ Location\ &\ Event\ Data\ Project\ -\ ACLED
	python $< 


data/acled-data3.csv: data_processing/acled-South-Sudan-Ethiopia.py $(RAW)/data_for_november_2019_evaluation/south_sudan_data/Humanitarian\ Data\ Exchange
	python $< 

data/dssat_data_oct2019_eval.tsv: data_processing/process_dssat_data_oct2019_eval.py
	python $< $@

data/IOM-DTM-data1.csv: data_processing/IOM-DTM1.py
	python $< $@

# =============================================================================
# Creation of final indicator table and flat list of indicators
# =============================================================================

data/indicator_data.tsv: data_processing/combine_data.py\
						 data/fao_data.tsv\
						 data/WHO-data1.csv\
						 data/WHO-data2.csv\
						 data/WHO-data3.csv\
						 data/IMF-data.csv\
						 data/WFP-data.csv\
						 data/World-Bank-data.csv\
						 data/acled-data1.csv\
						 data/acled-data2.csv\
						 data/acled-data3.csv\
						 data/IOM-DTM-data1.csv\
						 data/dssat_data_oct2019_eval.tsv\
						 $(RAW)/wm_12_month_evaluation/indicator_data_conflict.tsv\
						 data/indicator_data_fewsnet.tsv\
						 data/indicator_data_climis_unicef_ieconomics.tsv\
						 data/indicator_data_dssat.tsv\
						 data/indicator_data_UNHCR.tsv\
						 data/south_sudan_migration_data_secondary.tsv
	python $< $@

# =============================================================================
# Creation of concept to indicator mapping
# 
# *Please read before running Makefile*
# Instructions: You must clone/download the master branch from https://github.com/clulab/eidos
# and set the environment variable EIDOS to point to the local copy of the
# repository on your computer.
# 
# Please also read their instructions for the proper configuration and setup of 
# Eidos. scala, java, and sbt must be downloaded and installed. The most
# current versions of scala and sbt should work, however you cannot use the
# current version of java. openjdk version 1.8 is known to work with
# this build. You will need to reconfigure the below paths in accordance to
# where you downloaded/cloned Eidos. You must run "sbt assembly" in "/eidos"
# before running this portion of the Makefile. 
#
# Also, make sure to set 'useW2V' = true in $(EIDOS)/src/main/resources/eidos.conf 
#
# WARNING: EIDOS download and setup requires initially approximately 26.94 GB,
# however their instructions indicates options for optimizing and saving space.
# =============================================================================

EIDOS_SRC := $(EIDOS)/src/main

# Script that creates a YAML ontology from a flat list of variables gathered
# from our data tables.
ONTOLOGY_GEN_SCRIPT := $(EIDOS_SRC)/python/mk_yaml_ontology.py

TARGET_ONTOLOGY_FILENAME := $(EIDOS_SRC)/resources/org/clulab/wm/eidos/english/ontologies/delphi_db_indicators.yml
TARGET_ONTOLOGY_NAME := delphi_db_indicators

SCALA_VERSION:=2.12
EIDOS_VERSION:=0.2.3

WORD_EMBEDDING_VECTORS:=$(EIDOS)/src/main/resources/org/clulab/wm/eidos/english/w2v/glove.840B.300d.txt

$(WORD_EMBEDDING_VECTORS):
	cd $(@D); curl -O http://vision.cs.arizona.edu/adarsh/data/$(@F).gz; gunzip $(@F).gz

EIDOS_JAR:=target/scala-$(SCALA_VERSION)/eidos-assembly-$(EIDOS_VERSION)-SNAPSHOT.jar

$(EIDOS)/$(EIDOS_JAR):
	cd $(EIDOS); JAVA_OPTS=-Xmx50g time sbt assembly

data/primary_to_indicators.tsv: data/indicator_flat_list.txt\
						   $(ONTOLOGY_GEN_SCRIPT)\
						   data/indicator_data.tsv\
						   $(EIDOS)/$(EIDOS_JAR)\
						   $(WORD_EMBEDDING_VECTORS)
	python $(ONTOLOGY_GEN_SCRIPT) $< $(TARGET_ONTOLOGY_FILENAME) $(TARGET_ONTOLOGY_NAME) 
	cd $(EIDOS); time java -Xmx50g -cp $(EIDOS_JAR) org.clulab.wm.eidos.apps.CacheOntologies
	cp $(EIDOS_SRC)/resources/org/clulab/wm/eidos/english/ontologies/$(@F) $@


# =============================================================================
# Creation of reference evaluation CAGs
# =============================================================================

# Via scripting ---------------------------------------------------------------

build:
	mkdir -p $@


#Set SCENARIO to evaluations for precipitation build and hackathon for human migration build.
SCENARIO := hackathon

build/all_sts.pkl: $(SCENARIO)/create_all_sts.py $(PREASSEMBLED_CORPUS) 
	python $^ $@

build/reference_cag.pkl: $(SCENARIO)/create_reference_cag.py build/all_sts.pkl
	python $^ $@

build/scenario_centered_CAG.pkl: $(SCENARIO)/create_scenario_centered_CAG.py\
													  build/reference_cag.pkl\
													  data/delphi.db
	python $^ $@

build/CAG_with_indicators.pkl: $(SCENARIO)/create_CAG_with_indicators.py\
										 build/scenario_centered_CAG.pkl\
										 data/delphi.db
	python $< build/scenario_centered_CAG.pkl $@

build/parameterized_CAG.pkl: $(SCENARIO)/create_parameterized_CAG.py\
							 build/CAG_with_indicators.pkl\
							 data/delphi.db
	python $^ $@

build/quantified_CAG.pkl: $(SCENARIO)/create_quantified_CAG.py build/parameterized_CAG.pkl
	python $^ $@

report.pdf: report.tex apnotes.cls build/quantified_CAG.pkl
	latexmk -lualatex $<
	latexmk -c $(*F)

insert_model: $(SCENARIO)/insert_model_into_db.py build/quantified_CAG.pkl data/delphi.db
	python $^

build/delphi_cag.json: $(SCENARIO)/export_to_sri_and_cra.py build/quantified_CAG.pkl
	python $^ $@

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

.INTERMEDIATE: $(RAW_ZIP) $(RAW)/FAOSTAT $(RAW)/WDI $(RAW)/FEWSNET
