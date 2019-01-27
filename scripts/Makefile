all: delphi.db

delphi.db: data/south_sudan_data.tsv create_delphi_db.py
	python create_delphi_db.py

data:
	mkdir -p $@

# Intermediate files that will be deleted after downloading
.INTERMEDIATE: data/WDI_csv.tar.bz2 \
			   data/FAOSTAT.zip\
			   data/FAOSTAT\
			   data/FEWSNET/ALL_HFIC.zip\
			   data/FEWSNET/FEWSNET_World_Admin.zip\
			   fews

data/WDI_csv.tar.bz2: | data
	curl http://vision.cs.arizona.edu/adarsh/$(@F) -o $@

data/WDI_csv/WDIData.csv: data/WDI_csv.tar.bz2
	tar -C $(<D) -jxvf $<

data/FAOSTAT.zip: | data
	curl http://fenixservices.fao.org/faostat/static/bulkdownloads/$(@F) -o $@

data/FAOSTAT: data/FAOSTAT.zip
	unzip $< -d $@
	cd $@; ls *.zip | parallel --bar -k unzip 

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

data/south_sudan_data_fao.tsv: data/FAOSTAT
	python process_FAO_and_WDI_data.py --fao

data/south_sudan_data_wdi.tsv: data/WDI_csv/WDIData.csv
	python process_FAO_and_WDI_data.py --wdi

data/south_sudan_data_fewsnet.tsv: process_FEWSNET_data.py fews
	python $<

data/south_sudan_data_climis_unicef_ieeconomics.tsv: evaluations/month12/data_cleaning.py
	python $<

data/south_sudan_data_conflict.tsv:
	curl http://vision.cs.arizona.edu/adarsh/$(@F) -o $@

data/south_sudan_data.tsv: data/south_sudan_data_fao.tsv\
						   data/south_sudan_data_wdi.tsv\
						   data/south_sudan_data_conflict.tsv\
						   data/south_sudan_data_fewsnet.tsv\
						   data/south_sudan_data_climis_unicef_ieeconomics.tsv

	python process_FAO_and_WDI_data.py --combine



clean: 
	rm -rf data/FAOSTAT
	rm -rf data/WDI_csv
