all: | data/south_sudan_data.tsv

data:
	mkdir data

data/WDI_csv.tar.bz2: | data
	curl http://vision.cs.arizona.edu/adarsh/WDI_csv.tar.bz2 -o data/WDI_csv.tar.bz2

data/WDI_csv/WDIData.csv: | data/WDI_csv.tar.bz2
	cd data; tar -jxvf WDI_csv.tar.bz2

data/FAOSTAT.zip: | data
	curl http://fenixservices.fao.org/faostat/static/bulkdownloads/FAOSTAT.zip -o data/FAOSTAT.zip

data/FAOSTAT/*.zip: | data/FAOSTAT.zip
	cd data; unzip FAOSTAT.zip -d FAOSTAT

data/FAOSTAT/%.csv: data/FAOSTAT/%.zip
	cd data/FAOSTAT; unzip '$(<F)'

data/south_sudan_data_fao.tsv: | data/FAOSTAT/*.zip
	python process_FAO_and_WDI_data.py --fao

data/south_sudan_data_wdi.tsv: data/WDI_csv/WDIData.csv
	python process_FAO_and_WDI_data.py --wdi

data/south_sudan_data.tsv: data/south_sudan_data_fao.tsv data/south_sudan_data_wdi.tsv
	python process_FAO_and_WDI_data.py --combine
