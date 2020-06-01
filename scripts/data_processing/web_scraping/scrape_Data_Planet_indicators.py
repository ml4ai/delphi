import time
from selenium import webdriver, common
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import pandas as pd
import pickle


def scroll_to_element(browser, parent_id, element_id, topPos):
    if topPos == -1:
        java_script_call =\
            f"var myElement = document.getElementById('{element_id}');\
            var topPos = myElement.offsetTop;\
            document.getElementById('{parent_id}').scrollTop = topPos - 55;\
            console.log(topPos);\
            console.log(myElement.innerText);"
    else:
        java_script_call =\
            f"document.getElementById('{parent_id}').scrollTop = {topPos} - 55;\
            console.log({topPos});"
    #print(java_script_call)
    browser.execute_script(java_script_call)


options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')
# To open a browser window and watch the action comment the line below
#options.add_argument('--headless')

url_base = 'https://statisticaldatasets.data-planet.com/'

indicators = []
next_link_idx = 0

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

# Expand the tree view of the indicators
tree_pane = driver.find_element_by_id('isc_1C')

'''
with open('DP_branches.data', 'rb') as filehandle:
    # store the data as binary data stream
    branches = pickle.load(filehandle)
'''
branches = []

box_idx = 0
continuous_unsuccess = 0

while True:
    box_idx += 1
    if box_idx > 20000:
        exit()
        break

    table = driver.find_element_by_id('isc_1Ctable')
    rows = table.find_elements_by_css_selector('#isc_1Ctable > tbody > tr')
    num_lines = len(rows)

    try:
        tree_open_boxes = driver.find_element_by_id(f'isc_19open_icon_{box_idx}')
        tree_open_boxes.click()
        print(box_idx, 'GOOD')
        continuous_unsuccess = 0
        branches.append(box_idx)
        time.sleep(1)

        table = driver.find_element_by_id('isc_1Ctable')
        rows = table.find_elements_by_css_selector('#isc_1Ctable > tbody > tr')

        for row in rows:
            line_struct = row.find_elements_by_css_selector('td > div > table > tbody > tr > td')
            last_ID = line_struct[2].get_attribute('id')
            expanded_box_idx = int(last_ID.split('Cell')[1])
            if expanded_box_idx == box_idx:
                driver.execute_script(f"document.getElementById('isc_TreeViewImpl_1_0_body_cell{expanded_box_idx}_0').style.position = 'relative';")
                indicator = line_struct[2].get_attribute('innerText')
                offset_top = int(driver.find_element_by_id(f'isc_TreeViewImpl_1_0_body_cell{box_idx}_0').get_attribute('offsetTop'))
                print('Expanded Row: ', box_idx, last_ID, indicator, offset_top)
                break

        scroll_to_element(driver, 'isc_1C', f'isc_TreeViewImpl_1_0_body_cell{box_idx}_0', offset_top)
        time.sleep(1)
    except common.exceptions.NoSuchElementException:
        continuous_unsuccess += 1
        if continuous_unsuccess > num_lines:
            last_visible_line_struct = rows[-3].find_elements_by_css_selector('td > div > table > tbody > tr > td')
            last_ID = last_visible_line_struct[2].get_attribute('id')
            box_idx = int(last_ID.split('Cell')[1])
            driver.execute_script(f"document.getElementById('isc_TreeViewImpl_1_0_body_cell{box_idx}_0').style.position = 'relative';")
            indicator = last_visible_line_struct[2].get_attribute('innerText')
            offset_top = int(rows[-3].find_element_by_id(f'isc_TreeViewImpl_1_0_body_cell{box_idx}_0').get_attribute('offsetTop'))
            print(box_idx, last_ID, indicator, offset_top)
            continuous_unsuccess = 0
            scroll_to_element(driver, 'isc_1C', f'isc_TreeViewImpl_1_0_body_cell{box_idx}_0', offset_top)
            time.sleep(1)
        pass
    except common.exceptions.ElementNotInteractableException:
        pass
    except common.exceptions.StaleElementReferenceException:
        print('Done')
        break

exit()

with open('DP_branches.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(branches, filehandle)


tree_pane.send_keys(Keys.HOME)

'''
# Changing the zoom level of the browser
print('went home')
time.sleep(1)
driver.execute_script("document.body.style.zoom='150%'")
#ActionChains(driver).key_down(Keys.CONTROL).send_keys('-').key_up(Keys.CONTROL).perform()
print('size reduced')
time.sleep(15)
print('quitting')
driver.quit()
exit()
'''

first_ind_in_previous_page = ''
first_ind_in_current_page = ''
page = 1

#for page in range(5):
while True:
    table = driver.find_element_by_id('isc_1Ctable')
    rows = table.find_elements_by_css_selector('#isc_1Ctable > tbody > tr')
    for idx, row in enumerate(rows):
        level = row.get_attribute('aria-level')

        ind_struct = row.find_elements_by_css_selector('td > div > table > tbody > tr > td')
        #indicator = ind_struct[2].text
        indicator = ind_struct[2].get_attribute('innerText')
        ind_ID = ind_struct[2].get_attribute('id')

        # Below is the idea call that I wanted to make. However for ind_idx values
        # that end with 0, 1, 10 etc this removes the whole string :(
        #ind_idx = int(ind_ID.lstrip('isc_TreeViewImpl_1_0_valueCell'))
        try:
            ind_idx = int(ind_ID.split('isc_TreeViewImpl_1_0_valueCell')[1])
        except AttributeError:
            print('\t', 'AttributeError')
            ind_idx = -1

        print(idx, level, ind_idx, indicator)

        indicators.append({
                            'Page': page,
                            'idx': idx,
                            'ind_idx': ind_idx,
                            'Level': level,
                            'Indicator': indicator
                         })

        if idx == 0:
            first_ind_in_current_page = indicator

    if first_ind_in_previous_page == first_ind_in_current_page:
        break
    else:
        page += 1
        first_ind_in_previous_page = first_ind_in_current_page


    pd.DataFrame(indicators).to_csv('DP_indicator_list.csv', index=False)
    tree_pane.send_keys(Keys.PAGE_DOWN)
    time.sleep(1)

print('-----DONE-----')
driver.quit()
exit()
