import time
from selenium import webdriver
from bs4 import BeautifulSoup
import urllib.request
import json
import pandas as pd

limit = 50
last_offset = 4450

datasets = []
process_json_files = True

def extract_information_from_json(json_obj, dsets, start_idx):
    for db_idx, db_struct in enumerate(db_json['result'][0], start_idx):
        db_id = db_struct.get('id', '').strip()
        db_name = db_struct.get('title', '').strip()
        db_subtitle = db_struct.get('subtitle', '').strip()
        db_topics = db_struct.get('topic', '').strip().split(',')
        db_tags = db_struct.get('tags', '').strip().split(',')
        db_parent = db_struct.get('parent_catalog_title', '').strip()
        db_source = db_struct.get('source', '').strip()
        db_desc = db_struct.get('body', '').strip()
        db_type = db_struct.get('type', '').strip()
        db_spatial_granularity = db_struct.get('granularity', '').strip()
        db_temporal_coverage = db_struct.get('temporal_coverage', '').strip().split('-')
        #db_ = db_struct['']

        db_spatial_granularity = '' if db_spatial_granularity == 'Granularity not specified' else db_spatial_granularity

        dsets.append({
                        'ID': db_id,
                        'Name': db_name,
                        'Subtitle': db_subtitle,
                        'Topics': db_topics,
                        'Tags': db_tags,
                        'Parent': db_parent,
                        'Source': db_source,
                        'Description': db_desc,
                        'Type': db_type,
                        'Spatial Granularity': db_spatial_granularity,
                        'Temporal Coverage': db_temporal_coverage
                        })

        print(db_idx, db_name)

    #offset += 1
    #start_idx = db_idx + 1

    pd.DataFrame(dsets).to_csv(f'WB_datasets.csv', index=False)


if process_json_files:
    for offset in range(0, last_offset+1, limit):
        db_json = json.loads(open(f'Final_Extracted_Data/World_Bank/datasets_json/datasets_{offset}.json', 'r').read())

        extract_information_from_json(db_json, datasets, offset+1)
else:
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--incognito')
    # To open a browser window and watch the action comment the line below
    options.add_argument('--headless')

    url_base = 'https://datacatalog.worldbank.org/api/3/action/current_package_list_with_resources?format=json'

    offset = 0

    while True:
        # Open the web page with the topics
        driver = webdriver.Chrome(chrome_options=options)
        driver.get(f'{url_base}&limit={limit}&offset={offset}')
        time.sleep(3)

        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'lxml')
        driver.close()

        pre_element = soup.find('pre')

        db_json = json.loads(pre_element.get_text(), encoding='UTF-8')
        #print(json.dumps(db_json, sort_keys=True, indent=4, separators=(',', ': ')))

        if len(db_json['result'][0]) <= 0:
            print('******* Done ******')
            break

        with open(f'Final_Extracted_Data/World_Bank/datasets_{offset}.json', 'w') as f:
            f.write(pre_element.get_text())

        extract_information_from_json(db_json, datasets, offset+1)

        offset += limit

    '''
    try:
        with urllib.request.urlopen(f'{url_base}{offset}') as f:
           response = f.read()
    except urllib.error.HTTPError as err:
        print('EXCEPTION')
        print('\t', err)

    response_text = response.decode('utf-8')

    #db_json = json.loads(open('Final_Extracted_Data/World_Bank/databases_sample.json', 'r').read())
    db_json = json.loads(response_text, encoding='UTF-8')
    '''
