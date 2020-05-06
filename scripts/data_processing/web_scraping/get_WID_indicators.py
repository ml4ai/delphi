'''
    Download the indicator data table from:
    https://wid.world/summary-table/
    It is in excel format. Convert it to csv
    That csv is the input read by this script
'''

import pandas as pd

usecols = ['Short name of variable', 'Type(s) of variable', 'Years', 'Simple description of variable', 'Technical description of variable', 'Source']
ind_data = pd.read_csv('Final_Extracted_Data/WID/WID_SummaryTable_24May2019.csv', usecols=usecols)
ind_data.sort_values(by='Short name of variable', inplace=True)
#print(ind_data)

ind_data_grp = ind_data.groupby(by='Short name of variable')

indicators = []

for ind, grp in ind_data_grp:
    print(ind)
    ind_name = ind
    ind_type = grp['Type(s) of variable'].iloc[0]
    ind_simp_desc = grp['Simple description of variable'].iloc[0]
    ind_tech_desc = grp['Technical description of variable'].iloc[0]

    year_max = 0
    year_min = 3000
    for row in grp.iterrows():
        ind_years = row[1]['Years'].split('-')

        if len(ind_years) == 1:
            year_start = 3000
            year_end = int(ind_years[0])
        else:
            year_start = int(ind_years[0])
            year_end = int(ind_years[1])

        if year_start < year_min:
            year_min = year_start

        if year_end > year_max:
            year_max = year_end

    print('\t', year_min, year_max)

    indicators.append({
                        'Indicator': ind_name,
                        'Type': ind_type,
                        'Start Year': year_min if year_min != 3000 else '',
                        'End Year': year_max,
                        'Simple Description': ind_simp_desc,
                        'Technical Description': ind_tech_desc
                     })


pd.DataFrame(indicators).to_csv('WID_indicators.csv', index=False)
