all: | data/south_sudan_data.tsv fews

data:
	mkdir $@

data/WDI_csv.tar.bz2: | data
	curl http://vision.cs.arizona.edu/adarsh/$(@F) -o $@

data/WDI_csv/WDIData.csv: | data/WDI_csv.tar.bz2
	cd $(<D); tar -jxvf $(<F)

data/FAOSTAT.zip: | data
	curl http://fenixservices.fao.org/faostat/static/bulkdownloads/$(@F) -o $@

data/FAOSTAT: | data/FAOSTAT.zip
	unzip $< -d $@

data/FAOSTAT/%.csv: | data/FAOSTAT/%.zip
	unzip $< -d $(@D); rm $<

data/FEWSNET/FEWSNET_World_Admin.zip: | data
	mkdir -p $(@D)
	curl http://shapefiles.fews.net.s3.amazonaws.com/ADMIN/$(@F) -o $@

data/FEWSNET/FEWSNET_World_Admin: | data/FEWSNET/FEWSNET_World_Admin.zip
	unzip $< -d $@

data/FEWSNET/ALL_HFIC.zip:
	curl http://shapefiles.fews.net.s3.amazonaws.com/$(@F) -o $@

data/FEWSNET/ALL_HFIC: data/FEWSNET/ALL_HFIC.zip
	unzip $<
	cd $@/East\ Africa; for f in EA2017*; do mv $$f $${f/EA/EA_}; done

fews: | data/FEWSNET/FEWSNET_World_Admin data/FEWSNET/ALL_HFIC

data/south_sudan_data_fao.tsv: | data/FAOSTAT
	python process_FAO_and_WDI_data.py --fao

data/south_sudan_data_wdi.tsv: | data/WDI_csv/WDIData.csv
	python process_FAO_and_WDI_data.py --wdi

data/south_sudan_data.tsv: data/south_sudan_data_fao.tsv data/south_sudan_data_wdi.tsv
	python process_FAO_and_WDI_data.py --combine

clean: 
	rm -rf data/FAOSTAT
	rm -rf data/WDI_csv
