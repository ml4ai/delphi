import urllib.request
import json
import pandas as pd


#ind_json = json.loads(open('Final_Extracted_Data/World_Bank/indicators.json', 'r').read())

indicators_per_page = 17500

url_base = 'https://api.worldbank.org/v2/indicator?format=json&per_page='

with urllib.request.urlopen(f'{url_base}{indicators_per_page}') as f:
   response = f.read()

response_text = response.decode('utf-8')
ind_json = json.loads(response_text, encoding='UTF-8')

indicators = []

for ind_idx, ind_struct in enumerate(ind_json[1], 1):
    ind_id = ind_struct['id'].strip()
    ind_name = ind_struct['name'].strip()
    ind_unit = ind_struct['unit'].strip()
    ind_src_id = ind_struct['source']['id'].strip()
    ind_src_name = ind_struct['source']['value'].strip()
    ind_desc = ind_struct['sourceNote'].strip()

    ind_topics = []
    for ind_topic in ind_struct['topics']:
        ind_topics.append(ind_topic['value'].strip())

    indicators.append({
                        'ID': ind_id,
                        'Name': ind_name,
                        'Unit': ind_unit,
                        'Source ID': ind_src_id,
                        'Source': ind_src_name,
                        'Description': ind_desc,
                        'Topics': ind_topics
                     })

    print(ind_idx, ind_name)
    if len(ind_topics) > 1:
        print('\t', ind_id)
        print('\t', ind_unit)
        print('\t', ind_src_id)
        print('\t', ind_src_name)
        print('\t', ind_desc)
        print('\t', ind_topics)

pd.DataFrame(indicators).to_csv('WB_indicators.csv', index=False)
