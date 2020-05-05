import pandas as pd
from hdx.data.dataset import Dataset
from hdx.hdx_configuration import Configuration
import pprint
import pickle


def print_dict(dictionary):
    for key, value in dictionary.items():
        print(key, '\t:\t', value)

Configuration.create(hdx_site='prod', user_agent='A_Quick_Example', hdx_read_only=True)


dataset_name = 'acled-conflict-data-for-africa-1997-lastyear'
#dataset_name = '2016-sahel-inform'
dataset = Dataset.read_from_hdx(dataset_name)
#pprint.pprint(dataset.__dict__)

indicators = []
resources = []

datasets = Dataset.get_all_datasets()

with open('HDX_datasets.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(datasets, filehandle)

for dataset in datasets:
    dset_title = dataset.data.get('title', '')
    dset_notes = dataset.data.get('notes', '')
    dset_name = dataset.data.get('name', '')

    print('Title: ', dset_title)
    print('-------------------------------------------------------')
    print('\tName : ', dset_name)
    print('\tNotes: ', dset_notes)


    dset_filetypes = dataset.get_filetypes()
    print('\nFile Types')
    print(dset_filetypes)

    dset_url = dataset.get_hdx_url()
    print('\nURL')
    print(dset_url)

    print('\nlocation')
    print(dataset.get_location())

    print('\nMaintainer')
    mnt_disp_name = dataset.get_maintainer().get('display_name', 'Null')
    mnt_fullname = dataset.get_maintainer().get('fullname', 'Null')
    print('\tDisplay Name: ', mnt_disp_name) 
    print('\tFull Name   : ', mnt_fullname)

    print('\nOrganization')
    org_url = dataset.get_organization().get('org_url', '')
    org_disp_name = dataset.get_organization().get('display_name', '')
    org_title = dataset.get_organization().get('title', '')
    print('\tUrl         : ', org_url)
    print('\tDisplay Name: ', org_disp_name)
    print('\tTitle       : ', org_title)

    print('\nResources')
    for resource in dataset.get_resources():
        res_name = resource.get('name', '')
        res_url = resource.get('url', '')
        res_download_url = resource.get('download_url', '')
        res_description = resource.get('description', '')

        print('\tName        : ', res_name)
        print('\tUrl         : ', res_url)
        print('\tDownload Url: ', res_download_url)
        print('\tDescription : ', res_description)
        print()

        resources.append({
                        'Dataset Title': dset_title,
                        'Dataset Name': dset_name,
                        'Resource Name': res_name,
                        'Dataset URL': dset_url,
                        'Resource URL': res_download_url,
                        'Resource Description': res_description,
                        'Dataset Tags': dset_tags
                        })

    dset_tags = dataset.get_tags()
    print('\nTags')
    print(dset_tags)

    indicators.append({
                    'Title': dset_title,
                    'Name': dset_name,
                    'URL': dset_url,
                    'Maintainer': mnt_fullname,
                    'Organization': org_title,
                    'Organization URL': org_url,
                    'Notes': dset_notes
                     })

    pd.DataFrame(indicators).to_csv('HDX_indicators.csv', index=False)
    pd.DataFrame(resources).to_csv('HDX_indicators_with_resources.csv', index=False)

'''

dataset_names = Dataset.get_all_dataset_names()

dataset_names_df = pd.DataFrame({'Dataset Name': dataset_names})

dataset_names_df.sort_values(by='Dataset Name') 
dataset_names_df.to_csv('hdx_dataset_names.csv', index=False) 
'''
