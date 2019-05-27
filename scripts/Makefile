# This Makefile has not been fully tested. Please report any errors to
# adarsh.pyarelal@gmail.com

all: report.pdf

DSSAT_DATA := data/raw/DSSAT/disaggregated_all_20190123_1451
DSSAT_AGGREGATED_DATA := data/raw/DSSAT/SSD_csv

data/delphi.db: create_delphi_db.py\
				data/south_sudan_data.tsv\
				data/un_to_indicators.tsv\
				data/adjectiveData.tsv\
				$(DSSAT_AGGREGATED_DATA)
	python $^ $@

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
# =============================================================================

EIDOS := ~/eidos
EIDOS_SRC := $(EIDOS)/src/main
ONTOLOGY_GEN_SCRIPT := $(EIDOS_SRC)/python/mk_yaml_ontology.py
TARGET_ONTOLOGY_FILENAME := $(EIDOS_SRC)/resources/org/clulab/wm/eidos/english/ontologies/mitre12_indicators.yml
TARGET_ONTOLOGY_NAME := MITRE12

data/un_to_indicators.tsv: data/indicator_flat_list.txt $(ONTOLOGY_GEN_SCRIPT) data/south_sudan_data.tsv
	python $(ONTOLOGY_GEN_SCRIPT) $< $(TARGET_ONTOLOGY_FILENAME) $(TARGET_ONTOLOGY_NAME) 
	cd $(EIDOS)/src/main/scala/org/clulab/wm/eidos/apps; time java -Xmx50g -cp target/scala-2.12/eidos-assembly-0.2.2-SNAPSHOT.jar org.clulab.wm.eidos.apps.CacheOntologies
	cp $(EIDOS_SRC)/resources/org/clulab/wm/eidos/english/ontologies/$(@F) $@


# =============================================================================
# Creation of reference evaluation CAGs
# =============================================================================

# Via scripting ---------------------------------------------------------------

build:
	mkdir -p $@

PREASSEMBLED_CORPUS := data/raw/wm_12_month_evaluation/wm_12_month_4_reader_20190118.json

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
