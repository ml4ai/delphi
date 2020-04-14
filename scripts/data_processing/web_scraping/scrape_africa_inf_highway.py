import time
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup
import pandas as pd

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

for topic in topics[:2]:
    topic_name = topic.get_text().strip()
    print(topic_name)

    # Indicator web pages for these topics 
    # have different formatting. Omit them for now
    if topic_name in ['Africa']:
        continue

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
        dataset_name = dataset_result_item.find('label', 'dataset-result-name')
        print('\t', dataset_name.get_text().strip())

        source = dataset_result_item.find('div', 'source').find_all('span')[1].get_text()
        license = dataset_result_item.find('div', 'license').find_all('span')[1].get_text()
        owner = dataset_result_item.find('div', 'owner').find_all('span')[1].get_text()
        select_dataset = dataset_result_item.find('a', 'select-dataset')['href']

        print('\t\t', source)
        print('\t\t', license)
        print('\t\t', owner)
        print('\t\t', select_dataset)

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
        accord = driver.find_elements_by_css_selector('div.accordion > h3.ui-accordion-header a')
        accord[0].click()
        accord[1].click()
        time.sleep(2)

        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'lxml')
        driver.close()

        dataset_data.add((dataset_name.get_text().strip(),
                            source, 
                            license,
                            owner,
                            f'{url_base}{select_dataset}'))

        # Get the list of all the indicator details
        dimension_members_list = soup.find_all('div', 'dimension-members-list')
        indicator_elements = dimension_members_list[1].findChildren('div', recursive=False)

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
                                        'Dataset' : dataset_name.get_text().strip(),
                                        'Indicator' : indicator_name
                                    })
    print('----------------------------------')

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
