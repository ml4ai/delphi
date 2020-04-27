import time
import urllib.request
import json
import pandas as pd
import pickle
import urllib.request
import pickle


client_id = 'EZj54KGFo3rzIvnLczrElvAitEyU28DGw9R73tif'
version = '0.9942249010261828'

url_base = 'https://dataportal.opendataforafrica.org/api/1.0'
url_base = 'http://knoema.com/api/1.0/meta/dataset'

datasets = pd.read_csv('AIH_datasets_query_for_indicators.csv')

indicators = []

access_denied_datasets = []

max_level = 5
levels = {'0': '', '1': '', '2': '', '3': '', '4': '', '5':''}

for dset_idx, dset_struct in datasets.iterrows():

    dset_name = dset_struct['Dataset Name']
    dset_id = dset_struct['Dataset ID']
    dset_dimension = dset_struct['Dimension']

    print(dset_idx, dset_name, dset_id, dset_dimension)

    if dset_dimension == 'ignore':
        # Dataset name is the indicator name
        indicators.append({
                        'Dataset ID': dset_id,
                        'Dataset Name': dset_name,
                        'Dataset Dimension': dset_dimension,
                        'hasData': True,
                        '0': '', '1': '', '2': '', '3': '', '4': '', '5': ''
                        })
        continue

    req = urllib.request.Request(f'{url_base}/{dset_id}/dimension/{dset_dimension}?version={version}&client_id={client_id}')
    #req = urllib.request.Request(f'{url_base}/{dset_id}/dimension/{dset_dimension}')

    # Just in case if we issue more than 50 requests within certain time interval
    # (duration of the interval is unknown), we will get HTTPError exception.
    # In that case wait extra long and then try issuing the same request again
    # so that we will not miss data for this Topic.
    # Repeat this extra waiting and re-issuing request until we get data for
    # this Topic.
    while True:
        try:
            with urllib.request.urlopen(req) as f:
                response = f.read()
            break 
        except urllib.error.HTTPError:
            print('Too many requests')
            time.sleep(60)

    #print(response.decode('utf-8'))
    ind_json = json.loads(response.decode('utf-8'), encoding='UTF-8')
    #print(json.dumps(ind_json, sort_keys=True, indent=4, separators=(',', ': ')))

    try:
        items_struct = ind_json['items']
    except TypeError:
        print('\tAccess Denied')
        access_denied_datasets.append(dset_name)
        time.sleep(5)
        continue

    for ind_idx, ind_struct in enumerate(items_struct, 1):
        ind_name = ind_struct['name']
        ind_hasData = ind_struct['hasData']
        ind_level = ind_struct['level']

        levels[f'{ind_level}'] = ind_name

        for level in range(ind_level + 1, max_level + 1):
            levels[f'{level}'] = ''

        #print('\t'*ind_level, ind_idx, ' -|- '.join(levels))
        #print('\t'*ind_level, ind_idx, ind_name)
        #print('\t', ind_hasData)
        #print('\t', ind_level)

        indicator_row = {
                        'Dataset ID': dset_id,
                        'Dataset Name': dset_name,
                        'Dataset Dimension': dset_dimension,
                        'hasData': ind_hasData
                        }
        indicator_row.update(levels)
        indicators.append(indicator_row)

    pd.DataFrame(indicators).to_csv('AIH_indicators.csv', columns=['Dataset ID', 'Dataset Name', 'Dataset Dimension', '0', '1', '2', '3', '4', '5', 'hasData'], index=False)

    with open('AIH_access_denied_datasets.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(access_denied_datasets, filehandle)

    # We need to slow down to prevent issuing too many requests to the server
    # If we issue more than 50 requests, server stops giving results
    time.sleep(5)
