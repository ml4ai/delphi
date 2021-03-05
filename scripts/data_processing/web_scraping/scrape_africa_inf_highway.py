import time
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup
import pandas as pd
import urllib.request
import json

url_base = 'https://dataportal.opendataforafrica.org'

options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')
# To open a browser window and watch the action comment the line below
options.add_argument('--headless')

# Open the web page with the topics
driver = webdriver.Chrome(chrome_options=options)
driver.get(f'{url_base}/data/#menu=topic')
time.sleep(1)

page_source = driver.page_source
soup = BeautifulSoup(page_source, 'lxml')
driver.close()

# Get all the topics under with AIH hosts data
topics = soup.find_all('div', class_='group-item topic')

indicator_data = []

dataset_data = set()

for topic in topics[2:3]:
    topic_name = topic.get_text().strip()
    print(topic_name)

    # Indicator web pages for these topics 
    # have different formatting. Omit them for now
    #if topic_name in ['Africa']:
    #    continue

    #print(f'{url_base}/data/#topic={topic_name.replace(" ", "+")}')

    # Get the web page that lists the datasets for the current topic
    driver = webdriver.Chrome(chrome_options=options)
    driver.get(f'{url_base}/data/#topic={topic_name.replace(" ", "+")}')
    time.sleep(2)

    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'lxml')
    driver.close()

    # Extract the list of data set details for this topic
    dataset_result_items = soup.find_all('div', 'dataset-result-item')

    for dataset_result_item in dataset_result_items:
        dataset_name = dataset_result_item.find('label', 'dataset-result-name').get_text().strip()
        source = dataset_result_item.find('div', 'source').find_all('span')[1].get_text()
        license = dataset_result_item.find('div', 'license').find_all('span')[1].get_text()
        owner = dataset_result_item.find('div', 'owner').find_all('span')[1].get_text()
        select_dataset = dataset_result_item.find('a', 'select-dataset')['href']

        print('\t', dataset_name)
        print('\t\t', source)
        print('\t\t', license)
        print('\t\t', owner)
        print('\t\t', select_dataset)

        dataset_data.add((dataset_name,
                            source, 
                            license,
                            owner,
                            f'{url_base}{select_dataset}'))

        # Load the web page that lists indicators within this data set
        driver = webdriver.Chrome(chrome_options=options)
        driver.get(f'{url_base}{select_dataset}')
        time.sleep(2)

        # The indicators page has an accordion menu.
        # The first item is Country and it is open.
        # The second item is Indicator, which is closed and that is what we want.
        # Get the elements for the accordion menu items
        # Click on the first item (Country) to close it
        # Click on the second item (Indicator) to open it
        '''
        accord = driver.find_elements_by_css_selector('div.accordion > h3.ui-accordion-header a')
        accord[0].click()
        accord[1].click()
        time.sleep(2)
        '''

        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'lxml')
        driver.close()

        # Get the list of all the indicator details
        '''
        dimension_members_list = soup.find_all('div', 'dimension-members-list')
        indicator_elements = dimension_members_list[1].findChildren('div', recursive=False)
        '''

        system_client_id = soup.find(id='systemClientId')['value']
        print(system_client_id)
        #dataset_id = soup.find(id='datasetId')['value']
        dataset_id_elem = soup.find(id='datasetId')
        if dataset_id_elem == None:
            print(f'**** {dataset_name} is restricted ****')
            continue
        dataset_id = soup.find(id='datasetId')['value']
        print(dataset_id)
        ind_key = 'indicators' if topic_name == 'AEO' else 'indicator'
        indicator_url = f'{url_base}/api/1.0/meta/dataset/{dataset_id}/dimension/{ind_key}?client_id={system_client_id}&page_id={dataset_id}'
        print(indicator_url)

        req = urllib.request.Request(indicator_url)
        with urllib.request.urlopen(req) as f:
           response = f.read()
        #print(response.decode('utf-8'))
        ind_json = json.loads(response.decode('utf-8'), encoding='UTF-8')
        #print(json.dumps(ind_json, sort_keys=True, indent=4, separators=(',', ': ')))

        ind_levels = ['', '', '', '', '', '']
        group_levels = ['', '', '', '', '', '']
        min_ind_level = 0
        for idx, ind in enumerate(ind_json['items']):
            if ind['hasData']:
                ind_levels[ind['level']] = ind['name']

                # Created a flattened indicator name be flattening the nested
                # indicator names
                indicator_name = ' | '.join(ind_levels[min_ind_level:ind['level']+1])

                '''
                indicator_data.append({'Topic' : topic_name,
                                        'Group' : group_name,
                                        'Dataset' : dataset_name,
                                        'Indicator' : indicator_name
                                    })
                '''

                #print('\t'*ind['level'], ind['level'], ' | '.join(ind_levels[min_ind_level:ind['level']+1]))
                print('\t'*ind['level'], ind['level'], ind['name'])
            else:
                # This is something that does not have data. It cold either be a group name
                # or an indicator without any data. Here we are trying to filter out those two.
                # If it is a group name, what comes immediately after it must have a 'level'
                # greater than the group name (since indicators within a group nests within it)
                # else, it is an indicator name without data
                print('\t'*ind['level'], ind['name'], ind['fields']['code-6'])
                '''
                if idx < len(ind_json['items']) - 1 and ind_json['items'][idx+1]['level'] > ind['level']:
                    # This is a group name

                    # Indicator names within this group starts one level above the level 
                    # of the group name
                    min_ind_level = ind['level'] + 1

                    group_levels[ind['level']] = ind['name']

                    # Create a flattened group name by flattening the group name hierarchy
                    group_name = ' | '.join(group_levels[:ind['level']+1])
                    print('\t'*ind['level'], ' | '.join(group_levels[:ind['level']+1]))
                else:
                    # This is an indicator without data
                    print('\t'*ind['level'], ind['name'])
                '''

        #pd.DataFrame(indicator_data).to_csv(f'AIH_indicators_{topic_name}_{dataset_name}.csv', index=False, columns=['Topic', 'Dataset', 'Group', 'Indicator'])
        #exit()
        continue

        for indicator in indicator_elements:
            # Indicators have a level. Level 0 indicators are just group labels.
            # Indicator levels create a hierarchy. Each higher level nests within
            # its immediately preceding lower level item
            # Flatten this hierarchy a bit and try to preserve the information
            # as much as possible.
            level_div = indicator.find('div')
            level = int(level_div['class'][0][1])
            #print(level)

            if level == 0:
                group_name = level_div.get_text().strip()
                print('\t\t\t', '\t'*level, group_name)
            else:
                name_part = level_div.find('span').get_text().strip()
                if level == 1:
                    l1 = name_part
                    indicator_name = l1
                elif level == 2:
                    l2 = f'{l1} | {name_part}'
                    indicator_name = l2
                else:
                    l3 = f'{l2} | {name_part}'
                    indicator_name = l3

                print('\t\t\t', '\t'*level, indicator_name) 

                indicator_data.append({'Topic' : topic_name,
                                        'Group' : group_name,
                                        'Dataset' : dataset_name,
                                        'Indicator' : indicator_name
                                    })
    print('----------------------------------')

exit()

dataset_data_list = []
for dataset in dataset_data:
    dataset_data_list.append({'Dataset Name' : dataset[0],
                                'Source' : dataset[1],
                                'License' : dataset[2],
                                'Owner' : dataset[3],
                                'Link' : dataset[4]
                            })

dataset_df = pd.DataFrame(dataset_data_list)
indicator_df = pd.DataFrame(indicator_data)

dataset_df.to_csv('AIH_datasets.csv', index=False, columns=['Dataset Name', 'Source', 'License', 'Owner', 'Link'])
indicator_df.to_csv('AIH_indicators.csv', index=False, columns=['Topic', 'Dataset', 'Group', 'Indicator'])
