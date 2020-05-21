import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd

url_base = 'http://hdr.undp.org/en/data'

options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')
# To open a browser window and watch the action comment the line below
options.add_argument('--headless')

driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=options)

driver.get(url_base)
time.sleep(3)

indicators = []

next_level0_idx = 1
next_level1_idx = 1

while True:
    level0_elm = driver.find_element_by_id('indSelectors')
    level0_options = level0_elm.find_elements_by_tag_name('option')

    level1_elm = driver.find_element_by_id('secondaryDrop')

    level1_id = level0_options[next_level0_idx].get_attribute('value')
    level0 = level0_options[next_level0_idx].text

    level0_options[next_level0_idx].click()

    level1_div = driver.find_element_by_id(level1_id)
    level1_options = level1_div.find_elements_by_tag_name('option')

    # Although command below should give the option text
    # it does not. Luckily there is a workaround
    #level1 = level1_option.text
    level1 = level1_options[next_level1_idx].get_attribute('innerHTML')

    level1_options[next_level1_idx].click()
    button_div = driver.find_element_by_id('chartopts')
    link = button_div.find_element_by_tag_name('a')
    url = link.get_attribute('href')
    button = driver.find_element_by_id('gotoind')

    button.click()
    #driver.get(href)
    time.sleep(15)

    subheader_elm = driver.find_element_by_css_selector('h3.page-subheader')
    subheader = subheader_elm.text

    header_div = driver.find_element_by_id('header')
    header_div_inner = header_div.find_element_by_tag_name('div')
    header = header_div_inner.text 

    source_span = header_div_inner.find_element_by_tag_name('span')
    source = source_span.text 

    header_div_inner_html = header_div_inner.get_attribute('innerHTML')

    header_parts = header_div_inner_html.split('<span class="source">') 
    source = header_parts[-1].split('</span>')[0]
    description = header_parts[0].split('<br>')[2:-1]

    print(level0)
    print('\t', next_level0_idx, next_level1_idx, level1, url)
    print('\t'*2, subheader)
    print('\t'*2, description)
    print('\t'*2, source)

    driver.back()
    time.sleep(5)

    indicators.append({
                        '0': level0,
                        '1': level1,
                        'Description': '\n'.join(description),
                        'Sub-header': subheader,
                        'URL': url,
                        'Source': source
                     })

    pd.DataFrame(indicators).to_csv('HDR_indicators.csv', index=False)

    next_level1_idx += 1
    if next_level1_idx == len(level1_options):
        next_level0_idx +=1
        next_level1_idx = 1
        if next_level0_idx == len(level0_options):
            print('----- DONE -----')
            break

driver.close()
