import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd

options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')
# To open a browser window and watch the action comment the line below
#options.add_argument('--headless')

url_base = 'https://www.heritage.org/index/excel' #/2020/index2020_data.xls'

topics = []
indicators = []
next_link_idx = 0

#driver = webdriver.Chrome(chrome_options=options)
driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=options)
# Open the web page with the topics

for year in range(2020, 1990, -1):
    print(year)
    driver.get(f'{url_base}/{year}/index{year}_data.xls')
    time.sleep(10)

driver.close()
