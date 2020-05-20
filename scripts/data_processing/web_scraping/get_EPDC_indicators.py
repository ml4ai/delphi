import time
from selenium import webdriver
import pandas as pd

options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')
# To open a browser window and watch the action comment the line below
options.add_argument('--headless')

url_base = 'https://www.epdc.org/topic.html'

topics = []
indicators = []
next_link_idx = 0

driver = webdriver.Chrome(chrome_options=options)
# Open the web page with the topics
driver.get(url_base)
time.sleep(3)

#for link in href_elms[1:]:
while True:
    topics_elm = driver.find_element_by_id('block-system-main')
    href_elms = topics_elm.find_elements_by_tag_name('a')

    link = href_elms[next_link_idx]

    topic = link.text
    url = link.get_attribute('href')
    print(topic, url)

    link.click()
    time.sleep(3)

    desc_elm = driver.find_element_by_css_selector('div.key-indicators-chart > div.country_landing_graph.country_landing_graph_left')
    desc_paras = desc_elm.find_elements_by_tag_name('p')

    paras = []
    for para in desc_paras:
        paras.append(para.text)

    ind_struct_elm = driver.find_element_by_id('edit-indicators-subitems')
    ind_elms = ind_struct_elm.find_elements_by_css_selector('label.option')
    for ind in ind_elms:
        indicator = ind.text
        print('\t', indicator)

        indicators.append({
                            '0': topic,
                            '1': indicator
                         })
    driver.back()
    time.sleep(3)

    topics.append({
                    'Topic': topic,
                    'URL': url,
                    'Description': '\n'.join(paras)
                 })

    pd.DataFrame(topics).to_csv('EPDC_topics.csv', index=False)
    pd.DataFrame(indicators).to_csv('EPDC_indicators.csv', index=False)

    next_link_idx += 1
    if next_link_idx == len(href_elms):
        break

driver.close()
