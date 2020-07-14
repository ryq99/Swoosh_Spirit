import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.support import ui
from selenium.webdriver.common.keys import Keys
from urllib.request import urlretrieve

class snkrs_img_scraper():
    '''

    '''
    def __init__(self, url='https://www.pinterest.com'):
        self.browser = webdriver.Chrome(executable_path='chromedriver.exe')
        self.browser.get(url=url)
        wait = ui.WebDriverWait(self.browser, 10)
        wait.until(lambda x: x.find_element_by_tag_name("body"))
        self.img_list = []
        self.df = pd.DataFrame()

    def login(self, username, password):
        '''
        Log in with pinterest user name and password (google account does not work)

        browser: object, Chrome webdriver instance
        username: email, any other type of username is not supported yet

        '''
        assert '@' in username
        if self.browser.current_url != "https://www.pinterest.com/login/?referrer=home_page":
            self.browser.get("https://www.pinterest.com/login/?referrer=home_page")
        wait = ui.WebDriverWait(self.browser, 10)
        wait.until(lambda x: x.find_element_by_tag_name("body"))
        email = self.browser.find_element_by_xpath("//input[@type='email']")
        pswd = self.browser.find_element_by_xpath("//input[@type='password']")
        email.send_keys(username)
        pswd.send_keys(password)
        pswd.submit()
        wait = ui.WebDriverWait(self.browser, 10)
        wait.until(lambda x: x.find_element_by_name("searchBoxInput"))
        print('Login Successful!')

    def search_for_product(self, keyword=None):
        '''
        # Search for the product, this is the way to change pages later.

        browser: object, Chrome webdriver instance
        keyword: str, the keyword for searching

        '''
        seeker = self.browser.find_element_by_name("searchBoxInput")
        seeker.send_keys(Keys.CONTROL + "a")
        seeker.send_keys(Keys.DELETE)
        seeker.send_keys(keyword, Keys.ENTER)
        wait = ui.WebDriverWait(self.browser, 10)
        wait.until(lambda x: x.find_element_by_tag_name('img'))

    def get_img_urls_and_names(self, max_num_scroll=None, num_img = None):
        '''
        get a list of image tags, in which image urls and names can be found
        :return:
        '''
        lenOfPage = self.browser.execute_script("window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
        match = False
        num_of_scroll = 0
        #
        while (match == False):
            lastCount = lenOfPage
            time.sleep(3)
            lenOfPage = self.browser.execute_script("window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
            num_of_scroll += 1
            img_tags = self.browser.find_elements_by_tag_name('img')
            img_count = 0
            for img_tag in img_tags:
                urls = img_tag.get_attribute('srcset')
                title = img_tag.get_attribute('alt')
                self.img_list.append([urls, title])
                img_count += 1
            print(len(self.img_list), 'image urls copped.')
            if (lastCount == lenOfPage):
                match = True
            if num_img:
                if img_count >= num_img:
                    match = True
            if max_num_scroll:
                if (num_of_scroll == max_num_scroll):
                    match = True
            wait = ui.WebDriverWait(self.browser, 3)
        return self.img_list

    def img_list_2_df(self, clear_df=False, profile_name='Ray'):
        '''
        collect image urls and names into a data frame
        :param path:
        :return:
        '''
        if (len(self.df) == 0) or (clear_df == True):
            self.df = pd.DataFrame(self.img_list, columns=['url', 'title'])
        else:
            self.df = pd.concat([self.df, self.img_list], axis=0)
        self.df = (self.df.loc[(self.df['title'] != profile_name) &
                               (self.df['title'].isnull() != True) &
                               (self.df['title'].str.strip() != '') &
                               (self.df['title'].str.strip() != 'Promoted by')].reset_index(drop=True)
                          .assign(title=lambda x: x['title'].str.strip())
                          .assign(x1=lambda x: x['url'].apply(lambda x: x.split(', ')[0][:-3]).str.strip())
                          .assign(x2=lambda x: x['url'].apply(lambda x: x.split(', ')[1][:-3]).str.strip())
                          .assign(x3=lambda x: x['url'].apply(lambda x: x.split(', ')[2][:-3]).str.strip())
                          .assign(x4=lambda x: x['url'].apply(lambda x: x.split(', ')[3][:-3]).str.strip())[['title', 'x1', 'x2', 'x3', 'x4']])
        self.df.to_csv('data/img_title_url_df.csv', index=False)
        return self.df

    def download_images(self, url_list=None, label_name=None):
        path = os.path.join('data/images', label_name)
        if not os.path.isdir(path):
            os.mkdir(path)
        for url in url_list:
            urlretrieve(url, os.path.join(path, url.split('/')[-1]))
        print('all images are downloaded!')
