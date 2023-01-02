# coding=ISO-8859-2
import re
import time

from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

wait_time = 7


def deEmojify(text):
    emoj = re.compile("["
                      u"\U0001F600-\U0001F64F"  # emoticons
                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                      u"\U00002500-\U00002BEF"  # chinese char
                      u"\U00002702-\U000027B0"
                      u"\U00002702-\U000027B0"
                      u"\U000024C2-\U0001F251"
                      u"\U0001f926-\U0001f937"
                      u"\U00010000-\U0010ffff"
                      u"\u2640-\u2642"
                      u"\u2600-\u2B55"
                      u"\u200d"
                      u"\u23cf"
                      u"\u23e9"
                      u"\u231a"
                      u"\ufe0f"  # dingbats
                      u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', text)


def getClassName(div):
    try:
        class_names = div.attrs['class']
    except:
        class_names = div.parent.attrs['class']
    result = ""
    for name in class_names:
        result += f"{name} "
    return result[:-1]


class SimpleScrapeTools:
    def __init__(self, driver, html_markers_dict):
        self.driver = driver
        self.html_markers_dict = html_markers_dict

    def accept_terms_and_conditions(self):
        try:
            accept_bt = self.driver.find_elements(By.XPATH,"//form//button")[1]
            if accept_bt is not None:
                accept_bt.click()
        except IndexError:
            # if this error is raised it means that the terms and conditions are already accepted
            pass

    def wait_for_element_and_click(self, by_what, search_texts, place_iterator=None):
        wait = WebDriverWait(self.driver, wait_time)
        if place_iterator is not None:
            # wait.until(EC.presence_of_element_located((by_what, search_texts[0])))
            # element = driver.find_elements_by_xpath(search_texts[0])[0]
            wait.until(EC.presence_of_element_located((by_what, search_texts)))
            try:
                element = self.driver.find_elements(by_what,search_texts)[place_iterator]
            except IndexError:
                return -1
        else:
            wait.until(EC.presence_of_element_located((by_what, search_texts)))
            element = self.driver.find_elements(by_what, search_texts)[0]
        if element is not None:
            # self.driver.execute_script("arguments[0].scrollIntoView();", element)
            actions = ActionChains(self.driver)
            actions.move_to_element(element).click().perform()

    def wait_for_element(self, by_what, search_text):
        wait = WebDriverWait(self.driver, wait_time)
        wait.until(EC.presence_of_element_located((by_what, search_text)))

    def scroll_down(self, how_much_wait_at_page_end=-1, is_search_result=False, max_seconds=-1, number_of_scrolls=-1):
        scrollbox_selector = f"//div[contains(@class,'{self.html_markers_dict['scrollable_div']}')]" # "//div[contains(@class,'m6QErb DxyBCb kA9KIf dS8AEf')]"
        index = 0
        if is_search_result:
            index = 1
        self.wait_for_element(By.XPATH, scrollbox_selector)
        scrollable_div = self.driver.find_elements(By.XPATH,
            scrollbox_selector)[index]
        stopped_scrolling_counter = 0
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        start = time.time()
        # inner_timer_start = -1
        while True:
            self.driver.execute_script(
                        'arguments[0].scrollTop = arguments[0].scrollHeight',
                        scrollable_div
                    )
            time.sleep(0.1)
            if how_much_wait_at_page_end != -1:
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if last_height == new_height:
                    stopped_scrolling_counter += 1
                    if stopped_scrolling_counter >= how_much_wait_at_page_end:
                        break
                else:
                    stopped_scrolling_counter = 0
                    last_height = new_height

            if max_seconds != -1:
                if time.time() - start > max_seconds:
                    break

                # new_height = self.driver.execute_script("return document.body.scrollHeight")
                # if last_height == new_height:
                #     if inner_timer_start == -1:
                #         inner_timer_start = time.time()
                #     if inner_timer_start != -1  and time.time() - inner_timer_start > 3:
                #         break
                # else:
                #     inner_timer_start = -1

            if number_of_scrolls > 0:
                number_of_scrolls -= 1
                if number_of_scrolls == 0:
                    break


    def start_scraping(self, url):
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 10)
            self.accept_terms_and_conditions()
        except:
            raise Exception("Invalid url")

    def navigate_back(self, how_many):
        for i in range(how_many):
            self.driver.back()
            time.sleep(1)
            self.accept_terms_and_conditions()

    def open_a_new_tab_and_switch(self, url):
        self.driver.execute_script(f'''window.open("{url}","_blank");''')
        p = self.driver.current_window_handle

        # get newest child window
        child_tab = self.driver.window_handles[-1]
        self.driver.switch_to_window(child_tab)

        return p
