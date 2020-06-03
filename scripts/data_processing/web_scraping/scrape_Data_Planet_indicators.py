import time
from selenium import webdriver, common
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import copy


def scroll_to_element(browser, parent_id, element_id):
    java_script_call =\
        f"var myElement = document.getElementById('{element_id}');\
        myElement.style.position = 'relative';\
        var topPos = myElement.offsetTop;\
        document.getElementById('{parent_id}').scrollTop = topPos - 55;\
        console.log(topPos);\
        console.log(myElement.innerText);"
    browser.execute_script(java_script_call)


options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')
# To open a browser window and watch the action comment the line below
#options.add_argument('--headless')

url_base = 'https://statisticaldatasets.data-planet.com/'

driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=options)
driver.maximize_window()
# Open the web page with the topics
driver.get(url_base)
time.sleep(3)

cookies = driver.find_element_by_css_selector('div.cc-compliance')
cookies.click()

indicators = []

time.sleep(15)
#popup = driver.find_element_by_css_selector('button.WFSTOJB')
#popup.click()
#time.sleep(1)

box_idx = 0

max_level = 7
levels = {'0': '', '1': '', '2': '', '3': '', '4': '', '5': '', '6': '', '7': ''}
indicators = []

while True:
    if box_idx > 20000:
        pd.DataFrame(indicators).to_csv('data_planet_indicators.csv', index=False)
        print(f'box_idx exceeded {box_idx-1}')
        driver.quit()
        exit()

    try:
        full_td = driver.find_element_by_id(f'isc_TreeViewImpl_1_0_body_cell{box_idx}_0')
        full_tr = full_td.find_element_by_xpath('..')
        click_box = full_td.find_elements_by_css_selector('nobr > span')
        ind_td = full_td.find_element_by_id(f'isc_TreeViewImpl_1_0_valueCell{box_idx}')
        expandable = click_box[0].get_attribute('id')
        indicator = ind_td.get_attribute('innerText')
        ind_level = int(full_tr.get_attribute('aria-level'))

        levels[f'{ind_level - 1}'] = indicator

        for level in range(ind_level, max_level + 1):
            levels[f'{level}'] = ''

        print(box_idx, ind_level, indicator)

        if ind_level > 1:
            indicators.append(copy.deepcopy(levels))

        if expandable:
            expanded = eval(full_tr.get_attribute('aria-expanded').capitalize())
            if not expanded:
                click_box[0].click()
                time.sleep(1)

        box_idx += 1

    except common.exceptions.NoSuchElementException:

        pd.DataFrame(indicators).to_csv('data_planet_indicators.csv', index=False)

        try:
            scroll_to_element(driver, 'isc_1C', f'isc_TreeViewImpl_1_0_body_cell{box_idx - 1}_0')
            time.sleep(1)
        except common.exceptions.JavascriptException:
            print('---Done---')
            driver.quit()
            exit()
