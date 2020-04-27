'''
    Get metadata from:
    website: Africa Information Highway
    url: https://dataportal.opendataforafrica.org

    Gets the dataset information under each topic
    using direct network calls the website uses
    to get the data to render.

    @author: Manujinda Wathugala
    @date: 2020-04-20
'''

import time
import pandas as pd
import urllib.request
import urllib.parse
import json
import pickle

client_id = 'EZj54KGFo3rzIvnLczrElvAitEyU28DGw9R73tif'
version = '0.9942249010261828'

url_base = 'https://dataportal.opendataforafrica.org/api/1.0'

# Get the list of Topics
params = {
        'filter':{'filter': 'dsbrowser'},
        'includeCustomMetadataHierarchy': 'true'
        }

data = urllib.parse.urlencode(params)
data = data.encode('utf-8')
#req = urllib.request.Request(f'{url_base}/data/field/topic?version={version}&client_id={client_id}', data)
req = urllib.request.Request(f'{url_base}/data/field/topic', data)
with urllib.request.urlopen(req) as f:
   response = f.read()

#print(response.decode('utf-8'))
topics_json = json.loads(response.decode('utf-8'), encoding='UTF-8')
#print(json.dumps(topics_json, sort_keys=True, indent=4, separators=(',', ': ')))


datasets = []
dimensions = set() # Keep track of all the unique dimensions across datasets
empty_topics = []  # Keep track of Topics without datasets

# For each topic get the list of datasets
for topic_idx, topic_struct in enumerate(topics_json, 1):
    topic = topic_struct['id']
    print(topic_idx, topic)
    print('------------------------------')

    # We need to slow down to prevent issuing too many requests to the server
    # If we issue more than 50 requests, server stops giving results
    time.sleep(4)

    params = {
            'topic': f'{topic}',
            'filter': 'dsbrowser',
            'customMetadataFields': 'null',
            'fetchDimensions': 'true'
            }

    data = urllib.parse.urlencode(params)
    data = data.encode('utf-8')
    #req = urllib.request.Request(f'{url_base}/meta/dataset?version={version}&client_id={client_id}', data)
    req = urllib.request.Request(f'{url_base}/meta/dataset', data)

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
    dset_json = json.loads(response.decode('utf-8'), encoding='UTF-8')
    #print(json.dumps(dset_json, sort_keys=True, indent=4, separators=(',', ': ')))

    if len(dset_json) == 0:
        # This Topic had no datasets. Take note of the topic
        empty_topics.append(topic)


    # Parse through the dataset details and extract interesting information
    for dset_idx, dset_struct in enumerate(dset_json, 1):
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
                        'Topic' : topic,
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



    pd.DataFrame(datasets).to_csv(f'AIH_datasets_per_topic.csv', index=False, columns=['Topic', 'Dataset ID', 'Dataset Name', 'Dataset Owner', 'Dataset ref', 'Dataset Source ID', 'Dataset Source Name', 'Dataset Dimension IDs', 'Dataset Dimension Names', 'Dataset Description', 'Dataset Source License Type', 'Dataset Source Terms of Use'])

    with open('AIH_empty_topics.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(empty_topics, filehandle)

    with open('AIH_dimensions.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(dimensions, filehandle)

