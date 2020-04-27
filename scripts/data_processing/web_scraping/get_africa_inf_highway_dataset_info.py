import urllib.request
import json
import pandas as pd
import pickle

client_id = 'EZj54KGFo3rzIvnLczrElvAitEyU28DGw9R73tif'
version = '0.9942249010261828'

url_base = 'https://dataportal.opendataforafrica.org/api/1.0'

params = {
        'fetchDimensions': 'true'
        }

data = urllib.parse.urlencode(params)
data = data.encode('utf-8')
#req = urllib.request.Request(f'{url_base}/meta/dataset?version={version}&client_id={client_id}', data)
req = urllib.request.Request(f'{url_base}/meta/dataset', data)

with urllib.request.urlopen(req) as f:
   response = f.read()

dset_json = json.loads(response.decode('utf-8'), encoding='UTF-8')
#print(json.dumps(dset_json, sort_keys=True, indent=4, separators=(',', ': ')))

datasets = []
dimensions = set() # Keep track of all the unique dimensions across datasets

for dset_idx, dset_struct in enumerate(dset_json, 1):
    print(dset_idx, dset_struct['id'])
    print('\t', dset_struct['name'])
    print('\t', dset_struct['ref'])
    print('\t', dset_struct['source']['name'])

    dset_name = dset_struct['name']
    dset_id = dset_struct['id']
    dset_owner = dset_struct['owner']
    dset_desc = dset_struct['description']
    dset_ref = dset_struct['ref']
    dset_src_name = dset_struct['source']['name']
    dset_src_id = dset_struct['source']['id']
    dset_src_license_type = dset_struct['source']['licenseTypeName']
    dset_src_terms_of_use = dset_struct['source']['termsOfUseLink']

    print(dset_idx, dset_name)
    print('\t', dset_id)
    print('\t', dset_owner)
    print('\t', dset_ref)
    print('\t', dset_src_id, '\t', dset_src_name)

    dim_ids = []
    dim_names = []

    for dim_idx, dim_struct in enumerate(dset_struct['dimensions'], 1):
        dim_id = dim_struct['id']
        dim_name = dim_struct['name']

        dim_ids.append(dim_id)
        dim_names.append(dim_name)
        dimensions.add(dim_name)

        print('\t', dim_idx, '\t', dim_id, '\t', dim_name)

    datasets.append({
                    'Topic' : 'From API',
                    'Dataset ID' : dset_id,
                    'Dataset Name' : dset_name,
                    'Dataset Owner' : dset_owner,
                    'Dataset ref' : dset_ref,
                    'Dataset Source ID' : dset_src_id,
                    'Dataset Source Name' : dset_src_name,
                    'Dataset Dimension IDs' : dim_ids,
                    'Dataset Dimension Names' : dim_names,
                    'Dataset Description' : dset_desc,
                    'Dataset Source License Type' : dset_src_license_type,
                    'Dataset Source Terms of Use' : dset_src_terms_of_use 
                    })



pd.DataFrame(datasets).to_csv(f'AIH_datasets_from_API.csv', index=False, columns=['Topic', 'Dataset ID', 'Dataset Name', 'Dataset Owner', 'Dataset ref', 'Dataset Source ID', 'Dataset Source Name', 'Dataset Dimension IDs', 'Dataset Dimension Names', 'Dataset Description', 'Dataset Source License Type', 'Dataset Source Terms of Use'])

with open('AIH_dimensions_from_API.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(dimensions, filehandle)

