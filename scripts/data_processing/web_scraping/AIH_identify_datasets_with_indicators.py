import pandas as pd

usecols = ['Dataset ID', 'Dataset Name', 'Dataset Dimension IDs']

dsets = pd.read_csv('AIH_datasets_from_API.csv', usecols=usecols)

dsets.drop_duplicates(inplace=True)

dsets.sort_values(by='Dataset Name', inplace=True)

def extract_indicator_dimension(dimensions_list):
    if 'indicators' in dimensions_list:
        return 'indicators'
    elif 'indicator' in dimensions_list:
        return 'indicator'
    else:
        print(dimensions_list)
        return ''

dsets['Indicator'] = dsets['Dataset Dimension IDs'].apply(extract_indicator_dimension)
dsets.sort_values(by=['Indicator', 'Dataset Name'], inplace=True)

dsets.to_csv('AIH_filtered_datasets.csv', index=False)

