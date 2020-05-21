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

level0_elm = driver.find_element_by_id('indSelectors')
level0_options = level0_elm.find_elements_by_tag_name('option')

level1_elm = driver.find_element_by_id('secondaryDrop')

indicators = []

for level0_option in level0_options[1:]:
    level1_id = level0_option.get_attribute('value')
    level0 = level0_option.text
    print(level0)

    level1_div = driver.find_element_by_id(level1_id)
    level1_options = level1_div.find_elements_by_tag_name('option')

    for level1_option in level1_options[1:]:

        # Although command below should give the option text
        # it does not. Luckily there is a workaround
        #level1 = level1_option.text
        level1 = level1_option.get_attribute('innerHTML')
        print('\t', level1)

        indicators.append({
                            '0': level0,
                            '1': level1
                         })

pd.DataFrame(indicators).to_csv('HDR_indicators.csv', index=False)
driver.close()
