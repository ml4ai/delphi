How to get the list of indicators offered at the Africa Information Highway website
URL: https://dataportal.opendataforafrica.org/

1. Run get_africa_inf_highway_dataset_info.py
    This produces AIH_datasets_from_API.csv, which contains dataset information

2. Run AIH_identify_datasets_with_indicators.py
    This produces AIH_filtered_datasets.csv, which identifies which datasets has a dimension called 'indicator' or 'indicators'

3. Some datasets does not have a dimension called 'indicator' or 'indicators'. Visit the web pages for those datasets and manually check the available dimensions and see if some of the dimensions have useful data. In that case, note those dimensions in the AIH_filtered_datasets.csv file. If a particular dataset has more than one useful dimension, copy that dataset details to multiple lines in the file. If a dataset does not have any useful data, delete the row for that dataset. For some datasets, the dataset name is the indicator name. For such datasets put 'ignore' as the dimension.

4. Run get_africa_inf_highway_indicators.py
    This goes through the list of datasets and dimensions provided in AIH_filtered_datasets.csv and extracts the indicator names and produces AIH_indicators.csv, which contains the indicator data we are interested in.
