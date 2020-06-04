'''
    Scrape indicator information form the Data Planet website
    URL: https://statisticaldatasets.data-planet.com/

    1. Run the script
    2. Wait till the script opens the browser
    3. Most of the times there will be a pop-up window on the browser
    4. Click the button of the pop-up to make it disappear.
    5. Then the script will run till the end of extraction.
    
    The html structure of a single line of indicator information:

    <tr tabindex="1279" aria-level="4" aria-expanded="true" aria-labelledby="isc_TreeViewImpl_1_0_valueCell14292 isc_1B">
            <td class="treefolder" id="isc_TreeViewImpl_1_0_body_cell14292_0">
                    <div>
                            <table class="treefolder">
                                    <colgroup>
                                            <col width="60px">
                                            <col width="23px">
                                            <col>
                                    </colgroup>
                                    <tbody>
                                            <tr>
                                                    <td class="treefolder">
                                                            <span>
                                                            </span>
                                                    </td>
                                                    <td class="treefolder">
                                                            <nobr>
                                                                    <span id="isc_19open_icon_14292">
                                                                    </span>
                                                                    <span id="isc_19icon_14292">
                                                                    </span>
                                                            </nobr>
                                                    </td>
                                                    <td id="isc_TreeViewImpl_1_0_valueCell14292" class="treefolder" _wfx_id="academic-library-statistics-2014-to-current">
                                                            Academic Library Statistics (2014 to Current)
                                                    </td>
                                            </tr>
                                    </tbody>
                            </table>
                    </div>
            </td>
    </tr>
'''

import time
from selenium import webdriver, common
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains
import pandas as pd
import copy


def scroll_to_element(browser, container_id, element_id):
    # When possible, the contents of the container is
    # scrolled down until the element is at the top
    # of the container or as far up as it could be scrolled
    #
    # The element has absolute positioning, which gives the
    # position relative to the page. As the branches get
    # expanded, the container grows beyond the boundary of
    # the page and we need the offsetTop of the element
    # with respect to the container. So we change the style
    # of the element to make its position relative to the
    # container, which solves this issue.
    java_script_call =\
        f"var element = document.getElementById('{element_id}');\
        element.style.position = 'relative';\
        var topPos = element.offsetTop;\
        document.getElementById('{container_id}').scrollTop = topPos - 55;\
        console.log(topPos);\
        console.log(element.innerText);"
    browser.execute_script(java_script_call)


options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--disable-notifications')
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

# Trying to click on the pop-up message.
# This did not work. So we cannot run this script headless.
# We have to run the script, observe the browser window
# opened by selenium, wait for the pop-up message and click
# the button on it to make it go away.
#popup = driver.find_element_by_css_selector('button.WFSTOJB')
#popup.click()
#time.sleep(1)

box_idx = 0
last_scroll_box_idx = -1

max_level = 7
levels = {'0': '', '1': '', '2': '', '3': '', '4': '', '5': '', '6': '', '7': ''}
indicators = []

while True:
    # Mainly for debugging purposes
    # If you want to terminate the loop early extracting only few indicators
    # set the value of the if condition to the number of indicators  you
    # want to extract. 16939 was the total number of lines when the script
    # was run for the first time
    '''
    if box_idx > 16939:
        pd.DataFrame(indicators).to_csv('data_planet_indicators.csv', index=False)
        print(f'box_idx exceeded {box_idx-1}')
        driver.quit()
        exit()
    '''

    try:
        # Get the element that represents one row
        full_td = driver.find_element_by_id(f'isc_TreeViewImpl_1_0_body_cell{box_idx}_0')
        full_tr = full_td.find_element_by_xpath('..')

        # Get the element that represent the indicator information
        ind_td = full_td.find_element_by_id(f'isc_TreeViewImpl_1_0_valueCell{box_idx}')
        indicator = ind_td.get_attribute('innerText')

        # Get the position of this indicator in the hierarchy of nesting
        ind_level = int(full_tr.get_attribute('aria-level'))

        levels[f'{ind_level - 1}'] = indicator

        for level in range(ind_level, max_level + 1):
            levels[f'{level}'] = ''

        print(box_idx, ind_level, indicator)

        # We know for sure the ind_level = 1 are group labels
        # So we do not need them to occupy separate rows
        if ind_level > 1:
            indicators.append(copy.deepcopy(levels))

        # Check whether this is a line where we can expand the tree.
        # click_box gets two span elements. The first span is the
        # one to click to open up the branch. This span has an id
        # only when the current row is expandable. Otherwise we get
        # an empty string.
        click_box = full_td.find_elements_by_css_selector('nobr > span')
        expandable = click_box[0].get_attribute('id')

        if expandable:
            # aria-expanded is either true or false. We convert the value 
            # returned to True or False. The eval converts to a boolean 
            # in python.
            expanded = eval(full_tr.get_attribute('aria-expanded').capitalize())

            if not expanded:
                # Extend this branch
                # When click_box[0].click() is used,sometimes (specially on Mac)
                # tooltip from the previous line occludes the click box raising
                # an exception. So we have to use action chains to first move
                # the mouse over the click box and then perform the click.
                actions = ActionChains(driver)
                actions.move_to_element(click_box[0])
                actions.click(click_box[0])
                actions.perform()
                #click_box[0].click()
                time.sleep(1)

        box_idx += 1

    except (common.exceptions.NoSuchElementException, common.exceptions.ElementNotInteractableException):
        # We have reached the last indicator row attached to the container element.
        # We have to scroll down the view within the container to access any other rows

        pd.DataFrame(indicators).to_csv('data_planet_indicators.csv', index=False)

        if last_scroll_box_idx == box_idx:
            # We have not scrolled.
            # This must be the last line.
            # So stop
            print('---Done---')
            driver.quit()
            exit()

        last_scroll_box_idx = box_idx

        try:
            scroll_to_element(driver, 'isc_1C', f'isc_TreeViewImpl_1_0_body_cell{box_idx - 1}_0')
            time.sleep(1)
        except common.exceptions.JavascriptException:
            # This exception should not happen
            # Kept it here for additional safety
            print('---Javascript Exception---')
            driver.quit()
            exit()
