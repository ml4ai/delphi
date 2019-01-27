all: delphi.db

delphi.db: data/south_sudan_data.tsv create_delphi_db.py
	python create_delphi_db.py

data:
	mkdir -p $@

# Intermediate files that will be deleted after downloading

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

data/south_sudan_data_climis_unicef_ieeconomics.tsv: process_climis_unicef_ieeconomics_data.py
	python $<

data/south_sudan_data_conflict.tsv:
	curl http://vision.cs.arizona.edu/adarsh/$(@F) -o $@

# =============================================================================
# Creation of final indicator table and flat list of indicators
# =============================================================================

data/south_sudan_data.tsv data/indicator_flat_list.txt: process_FAO_and_WDI_data.py\
														data/south_sudan_data_fao.tsv\
														data/south_sudan_data_wdi.tsv\
														data/south_sudan_data_conflict.tsv\
														data/south_sudan_data_fewsnet.tsv\
														data/south_sudan_data_climis_unicef_ieeconomics.tsv

	python $< --combine


# =============================================================================
# Creation of concept to indicator mapping
# =============================================================================

EIDOS := ~/clulab/eidos
EIDOS_SRC := $(EIDOS)/src/main
ONTOLOGY_GEN_SCRIPT := $(EIDOS_SRC)/python/mk_yaml_ontology.py
TARGET_ONTOLOGY_FILENAME := $(EIDOS_SRC)/resources/org/clulab/wm/eidos/english/ontologies/mitre12_indicators.yml
TARGET_ONTOLOGY_NAME := MITRE12

data/un_to_indicators.tsv: data/indicator_flat_list.txt $(ONTOLOGY_GEN_SCRIPT)
	
	python $(ONTOLOGY_GEN_SCRIPT) $< $(TARGET_ONTOLOGY_FILENAME) $(TARGET_ONTOLOGY_NAME) 
	cd $(EIDOS); time sbt "runMain org.clulab.wm.eidos.apps.CacheOntologies"


