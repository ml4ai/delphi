'''
    Extracts indicator information from:
    UNESCO Institute for Statistics (UIS)
    url: http://data.uis.unesco.org/

    For the time being:
    1. Load the homepage
    2. View page source
    3. Locate the ul element with class="treeview"
       under the div with id="browsethemes"
    4. Copy everything in that element into a text file
       called UIS_indicator_html.txt
    5. That text file is the input to this script

    Might be able to automate this process so that the
    script loads the homepage and extracts the element
    of interest and proceed.

    At the writing this script, I did not spend time 
    on that.
'''

from bs4 import BeautifulSoup
import pandas as pd

# At the time of writing this script there were
# 4 levels in the indicator tree hierarchy
max_levels = 4

def extract_indicator_text(level, level_li, level_text_lst, inds):

    if level > max_levels:
        print('There are more levels now. Increase max_levels and re-run')
        exit()

    level_span = level_li.select_one('span')

    if level_span:
        # This is a group label heading
        level_text_lst[level] = level_span.get_text()
        print('\t'*level, level_text_lst[level])

        #if level < max_levels:
        level_uls = level_li.findChildren('ul', recursive=False)
        level_lis = level_uls[0].findChildren('li', recursive=False)
        for next_level_li in level_lis:
            extract_indicator_text(level+1, next_level_li, level_text_lst, inds)
    else:
        level_a = level_li.select_one('a.q')

        if level_a:
            # This is an indicator with data
            level_text_lst[level] = level_a.get_text()
            print('\t'*level, '****', level_text_lst[level])

            for lvl in range(level+1, max_levels+1):
                level_text_lst[lvl] = ''

            row = {}
            for lvl in range(max_levels + 1):
                row[str(lvl)] = level_text_lst[lvl]

            row['URL'] = ''

            inds.append(row)
            '''
            inds.append({
                                '0': level_text_lst[0],
                                '1': level_text_lst[1],
                                '2': level_text_lst[2],
                                '3': level_text_lst[3],
                                '4': level_text_lst[4]
                             })
            '''
        else:
            level_a = level_li.select_one('a.le')
            if level_a:
                # This is a downloadable file
                level_text_lst[level] = level_a.get_text()
                print('\t'*level, '++++', level_text_lst[level])
                print('\t'*(level+1), '++++', level_a['href'])

                for lvl in range(level+1, max_levels+1):
                    level_text_lst[lvl] = ''

                row = {}
                for lvl in range(max_levels + 1):
                    row[str(lvl)] = level_text_lst[lvl]

                row['URL'] = level_a['href']

                inds.append(row)


indicators = []
level_text = ['', '', '', '', '']

with open('Final_Extracted_Data/UIS/UIS_indicator_html.txt', 'r') as ind_html:
    soup = BeautifulSoup(ind_html.read(), 'lxml')
    #print(soup.find(class_='treeview').prettify())
    #exit()
    level0_lis = soup.select("ul.treeview > li")
    print(len(level0_lis))
    for level0_li in level0_lis:
        extract_indicator_text(0, level0_li, level_text, indicators)

pd.DataFrame(indicators).to_csv('UIS_indicators.csv', index=False)
