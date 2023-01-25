# coding=ISO-8859-2
import re
import time
from datetime import timedelta, datetime
from typing import List

from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

from app.services.analysis import geolocation
from app.services.scraper.models.people_simple_credentials import PeopleSimpleCredentials
from app.services.scraper.models.position import Position
from app.services.scraper.tools import geocode_api


def getMissingUrl(reviewer_id, review_id):
    return f"https://www.google.com/maps/contrib/{reviewer_id}/place/{review_id}/"


def delete_unsupported_characters(type_of_object):
    try:
        print(type_of_object)
        return type_of_object
    except:
        place_type = type_of_object
        iterator = 0
        new_place_type = ""
        for letter in place_type:
            if ord(letter) == 8211:  # myślnik
                new_letter = '-'
            elif ord(letter) == 183:  # kropka
                new_letter = ''
            elif ord(letter) == 8222 or ord(letter) == 8221:  # otwarcie/zamknięcie cudzysłowia
                new_letter = '/"'
            else:
                new_letter = letter
            new_place_type += new_letter
        return new_place_type


def convert_from_relative_to_absolute_date(relative_date) -> datetime:
    relative_date = relative_date.replace("\xa0", " ")
    data = relative_date.split(' ')
    # unit is amount of days
    if len(data) == 1:
        value = 1
        unit = 'dzień'
    elif len(data) == 2:
        value = 1
        unit = data[0]
    elif len(data) == 3:
        value = int(data[0])
        unit = data[1]
    else:
        return None
    if unit == "godzin" or unit == "godzinę" or unit == "godziny" or unit == "minut" or unit == "minutę":
        unit = 0
    elif unit == "dzień" or unit == "dni" or unit == "dnia":
        unit = 1
    elif unit == "tydzień" or unit == "tygodnie":
        unit = 7
    elif unit == "miesiąc" or unit == "miesiące" or unit == "miesięcy":
        unit = 30
    elif unit == "rok" or unit == "lata" or unit == "lat":
        unit = 365
    time_delta = timedelta(value * unit)
    current_date = datetime.today()
    return current_date - time_delta


class InfoScrapeTools:
    def __init__(self, driver, simpleScrapeTools, html_markers_dict):
        self.driver = driver
        self.simpleScrapeTools = simpleScrapeTools
        self.html_markers_dict = html_markers_dict

    @staticmethod
    def get_number_of_reviews_from_split_text(split_text: List[str]) -> int:
        if len(split_text) >= 2 and split_text[1] == "opinia":
            number_of_reviews = 1
        elif len(split_text) >= 2 and (split_text[1] == "opinie" or split_text[1] == "opinii"):
            number_of_reviews = int(split_text[0])
        elif len(split_text) >= 3 and (split_text[2] == "opinie" or split_text[2] == "opinii"):
            number_of_reviews = int(split_text[0]) * 1000 + int(split_text[1])
        else:
            number_of_reviews = 1
        return number_of_reviews


    def get_number_of_reviews_of_place(self, response=None) -> int:
        if response is None:
            response = BeautifulSoup(self.driver.page_source, 'html.parser')
        try:  # if on place page
            review_element = response.find_all(jsaction='pane.rating.moreReviews')[0].text
            review_info = review_element[3:].split()
            # review_info = response.find_all(class_=re.compile(self.html_markers_dict["place_number_of_reviews"]))[
            #     1].text.split()  # "gm2-button-alt jqnFjrOWMVU__button-blue").text.split()
            return self.get_number_of_reviews_from_split_text(review_info)
        except:  # if on reviews page
            review_info = response.find(jsan="7.fontBodySmall,0.jslog").text.split()
            if len(review_info) > 2:
                number_of_reviews = int(review_info[1]) * 1000 + int(review_info[2])
            elif len(review_info) == 2 and review_info[1].isnumeric():
                number_of_reviews = int(review_info[1])
            else:
                number_of_reviews = 1
            return number_of_reviews

    def get_number_of_reviews_of_person(self, response=None):
        if response is None:
            response = BeautifulSoup(self.driver.page_source, 'html.parser')
        try:
            review_info = response.find(
                class_=re.compile(self.html_markers_dict["reviewer_number_of_reviews"])).text.split()
        except AttributeError:
            temp = self.driver.current_url
            return 15
        number_of_reviews = int(review_info[0])
        if len(review_info) >= 2 and review_info[1] == "opinia":
            number_of_reviews = 1
        elif len(review_info) >= 2 and (review_info[1] == "opinie" or review_info[1] == "opinii"):
            number_of_reviews = int(review_info[0])
        elif len(review_info) >= 3 and (review_info[2] == "opinie" or review_info[2] == "opinii"):
            number_of_reviews = int(review_info[0]) * 1000 + int(review_info[1])
        else:
            number_of_reviews = 1
        try:
            number_of_ratings = int(review_info[3])
            number_of_reviews += number_of_ratings
        except:
            pass
        return number_of_reviews

    def get_accounts_from_localization(self, accounts_list, limit=150):
        number_of_reviews = self.get_number_of_reviews_of_place()
        if number_of_reviews > limit:
            number_of_reviews = limit
        self.simpleScrapeTools.scroll_down(int(number_of_reviews / 8) + 2)
        response = BeautifulSoup(self.driver.page_source, 'html.parser')
        reviewer_names = response.find_all(class_=self.html_markers_dict["place_reviewer_name"])
        reviewer_url = response.find_all(class_=re.compile(self.html_markers_dict["place_reviewer_url"]))
        for (name, temp_url) in zip(reviewer_names, reviewer_url):
            name = str(name.text)[1:-1]
            temp_url = str(temp_url).split()[6][6:-1]
            accounts_list.append(PeopleSimpleCredentials(name, temp_url))

    def navigate_to_place_details(self):
        try:
            self.simpleScrapeTools.wait_for_element_and_click(By.XPATH,
                                                              f"//button[contains(@class,'{self.html_markers_dict['trans_more_info_button']}')]")
        except TimeoutException:
            self.simpleScrapeTools.wait_for_element_and_click(By.XPATH,
                                                              "//button[contains(@class,'section-place-name-header-place-details blue-button-text')]")  # constant and hard to

    def click_on_more_reviews(self):
        try:
            self.simpleScrapeTools.wait_for_element_and_click(By.XPATH,
                                                              "//button[contains(@jsaction,'pane.rating.moreReviews')]")
        except TimeoutException:
            self.simpleScrapeTools.wait_for_element_and_click(By.XPATH,
                                                              "//span[contains(@aria-label,'opin')]")
    def navigate_to_place_site(self, met_people, place_iterator):
        if self.simpleScrapeTools.wait_for_element_and_click(By.XPATH,
                                                             f"//div[contains(@class,'{self.html_markers_dict['reviewer_review_label']}')]",
                                                             place_iterator) == -1:
            return self.driver.current_url
        try:
            self.navigate_to_place_details()
        except TimeoutException:
            pass
        finally:  # some places go straight to the site (skipping the details panel)
            refresh = False
            try:
                self.simpleScrapeTools.wait_for_element_and_click(By.XPATH,
                                                                  "//button[@jsaction='pane.rating.moreReviews']")  # constant value for now
            except TimeoutException:
                self.driver.refresh()
                refresh = True
                try:
                    self.simpleScrapeTools.wait_for_element_and_click(By.XPATH,
                                                                      "//button[contains(@jsaction,'pane.rating.moreReviews')]")  # constant value for now
                except:  # some places don't have reviews (weird)
                    self.simpleScrapeTools.navigate_back(1)
                    return self.driver.current_url
        place_url = self.driver.current_url
        self.get_accounts_from_localization(met_people)

        if refresh:
            self.driver.back()
            time.sleep(1)
        self.simpleScrapeTools.navigate_back(2)
        return place_url

    def wait_for_person_to_load(self):
        try:
            self.simpleScrapeTools.wait_for_element(By.XPATH,
                                                    f"//*[@class='{self.html_markers_dict['reviewer_name']}']")  # 'reviewer_number_of_reviews']}']")
            return 0
        except TimeoutException:
            return -1

    def wait_for_place_site_to_load(self):
        try:
            self.simpleScrapeTools.wait_for_element(By.XPATH,
                                                    f"//*[@class='{self.html_markers_dict['place_name']}']")
            return 0
        except TimeoutException:
            return -1

    def is_private_account(self, response=None):
        if response is None:
            self.wait_for_person_to_load()
            response = BeautifulSoup(self.driver.page_source, 'html.parser')
        empty_account = response.find(class_=re.compile(self.html_markers_dict['reviewer_private_label']))
        if empty_account is None:
            return False
        else:
            return True

    def getCoordinatesList(self):
        self.wait_for_person_to_load()
        number_of_reviews = self.get_number_of_reviews_of_person()

        position_list = []

        self.simpleScrapeTools.scroll_down(int(number_of_reviews / 8) + 2)
        response = BeautifulSoup(self.driver.page_source, 'html.parser')
        for place_iterator in range(number_of_reviews):
            try:  # Throws exception if google response is unexpected
                position = self.get_reviewer_place_location(response, place_iterator)
                if geolocation.is_in_poland(position.lat, position.lon):
                    position_list.append(position)
            except IndexError:  # if the place doesnt have a location
                pass
        return position_list

    def getRatingsInfo(self, is_loaded=False):
        self.wait_for_person_to_load()
        response = BeautifulSoup(self.driver.page_source, 'html.parser')
        number_of_reviews = self.get_number_of_reviews_of_person()
        if not is_loaded:
            self.simpleScrapeTools.scroll_down(int(number_of_reviews / 8) + 2)
            response = BeautifulSoup(self.driver.page_source, 'html.parser')
        stars_list = []
        for place_iterator in range(number_of_reviews):
            try:
                stars_count = self.get_stars_count(response, place_iterator)
                stars_list.append(stars_count)
            except IndexError:  # if the place doesnt have a location or can't be accessed
                pass
        return stars_list

    def get_reviewer_place_location(self, response, place_iterator):
        address = response.find_all(class_=re.compile(
            self.html_markers_dict['reviewer_review_location']))[
            place_iterator][:-4]
        address = address.text
        return geocode_api.forward_geocode(address)

    def get_place_location(self, address):
        return geocode_api.forward_geocode(address)

    def getPlaceSpecificData(self, place_iterator):
        if self.simpleScrapeTools.wait_for_element_and_click(By.XPATH,
                                                             f"//div[contains(@class,'{self.html_markers_dict['reviewer_review_label']}')]",
                                                             place_iterator) == -1:
            return None
        try:
            self.navigate_to_place_details()
        except TimeoutException:
            pass
        type_of_object = None
        place_url = self.driver.current_url
        try:
            self.wait_for_place_site_to_load()
            response = BeautifulSoup(self.driver.page_source, 'html.parser')
            # type_of_object = \
            #     response.find(class_=re.compile("x3AX1-LfntMc-header-title")).find_all(class_="h0ySl-wcwwM-E70qVe")[1].find(
            #         class_=re.compile("widget-pane-link"))
            place_url = self.driver.current_url
            type_of_object = self.get_place_type(response)
            self.simpleScrapeTools.navigate_back(1)
        except:
            pass
        return {"place_url": place_url, "type_of_object": type_of_object}

    def get_stars_count(self, response, place_iterator):
        try:
            star_response = response.find_all(class_=re.compile(self.html_markers_dict['all_reviewer_stars']))[
                place_iterator]
            star_text = star_response.text
            stars_count = int(star_text.split()[0])
        except:  # some places have x/5 ratings not stars
            star_response = response.find_all(class_=re.compile(self.html_markers_dict['hotel_rating_label']))[
                place_iterator]
            stars_count = int(star_response.text.split('/')[0])
        return stars_count

    def get_place_type(self, place_response) -> str:
        try:
            return place_response.find(jsaction="pane.rating.category").text.strip()
        except:
            return "Hotel"

    def find_from_complicated_html_marker(self, html_marker, context, level_of_recursion = 0):
        if isinstance(html_marker, list):
            if level_of_recursion + 1 == len(html_marker):
                return context
            current_marker = html_marker[level_of_recursion]
            search1 = context.find(class_=re.compile(current_marker))
            return self.find_from_complicated_html_marker(html_marker, search1, level_of_recursion + 1)
        else:
            raise TypeError("html_marker must be a list")

    def find_using_html_marker(self, html_marker, context, multiple = False):
        if context is None:
            context = BeautifulSoup(self.driver.page_source, 'html.parser')
        if isinstance(html_marker, list):
            return self.find_from_complicated_html_marker(html_marker, context)
        else:
            if multiple:
                return context.find_all(class_=re.compile(html_marker))
            else:
                return context.find(class_=re.compile(html_marker))

    def find_review_response(self, context):
        review_response_markers = self.html_markers_dict['place_reviewer_response_content']
        search1 = context.find_all(class_=re.compile(review_response_markers[0]))
        search2 = search1[0].find_all(class_=re.compile(review_response_markers[1]))
        search3 = search2[1].find_all(class_=re.compile(review_response_markers[2]))
        search3 = search3[0].find_all(class_=re.compile(review_response_markers[3]))
        return search3[0].text

    # def get_reviewer_response(self, context) -> Optional[str]:
    #     html_markers = self.html_markers_dict['place_reviewer_response_content']
    #     search1 = context.find_all(class_=re.compile(html_markers[0]))
    #     for sub_div in search1.:
    #         if sub_div.find(class_=re.compile
    def find_and_get_absolute_date(self, response, place_iterator, relative_date = None) -> datetime:
        if relative_date is None:
            relative_date = response.find_all(class_=re.compile(self.html_markers_dict['reviewer_review_relative_date']))[
                place_iterator].text
        relative_date = relative_date.replace("\xa0", " ")
        return convert_from_relative_to_absolute_date(relative_date)

    def find_the_right_results(self, lat, lon):
        try:
            self.simpleScrapeTools.scroll_down(5, is_search_result=True)
            time.sleep(1)
            response = BeautifulSoup(self.driver.page_source, 'html.parser')
            places = response.find_all(
                class_=re.compile(self.html_markers_dict['search_result_marker_0'].split(' ')[0]))  # bXlT7b-hgDUwe
            if len(places) == 0:
                places = response.find_all(class_=re.compile(self.html_markers_dict['search_result_marker_1']))
            places = places[::2]  # two matches for one place
            distance_list = []
            desired_position = Position(lat, lon)
            for place in places:
                try:
                    child = place.find(class_=re.compile("bXlT7b-hgDUwe"))
                    # iterator += 1
                    # if iterator%3 != 1:
                    #     continue
                    parent = child.parent
                    address = parent.text[3:]
                    position = geocode_api.forward_geocode(address)
                    distance = geolocation.distance(position, desired_position)
                    distance_list.append(distance)
                except:
                    distance_list.append(100.0)
            if min(distance_list) < 0.5:
                index = distance_list.index(min(distance_list)) * 2  # there are two matches for one place
                place = self.driver.find_elements(By.XPATH, f"//*[contains(@class,"
                                                           f"'{self.html_markers_dict['search_result_marker_0'].split(' ')[0]}')]")[
                    index]
                place.click()
                return 0
            else:
                return -1
        except:
            return -1

    def search_for_potential_results(self, search_string, lat, lon, search_bar=None):
        if search_bar is None:
            search_bar = self.driver.find_elements(By.XPATH, f"//input[@id='searchboxinput']")[0]
        search_bar.clear()
        time.sleep(0.5)
        search_bar.send_keys(f"{lat}, {lon}")
        search_bar.send_keys(Keys.ENTER)
        time.sleep(1.5)
        search_bar.clear()
        try:
            time.sleep(0.5)
            search_bar.send_keys(search_string)
            search_bar.send_keys(Keys.ENTER)
            time.sleep(1.5)
        except:
            search_bar = self.driver.find_elements(By.XPATH, f"//input[@id='searchboxinput']")[0]
            search_bar.clear()
            time.sleep(1.5)
            search_bar.send_keys(search_string)
            search_bar.send_keys(Keys.ENTER)
            time.sleep(1.5)

        try:
            places = self.driver.find_elements(By.XPATH, f"//a[contains(@aria-label,'{search_string}')]")
            if len(places) > 1:
                raise Exception("More than one place with the given name")
            places[0].click()
        except:
            if self.wait_for_place_site_to_load() == -1:
                if self.find_the_right_results(lat, lon) == -1:
                    raise Exception("Did not go to the desired place")

    def search_in_google_maps_for(self, search_string):
        try:
            search_bar = self.driver.find_elements(By.XPATH, f"//input[@id='searchboxinput']")[0]
        except:
            self.simpleScrapeTools.start_scraping("https://www.google.pl/maps/")
            search_bar = self.driver.find_elements(By.XPATH, f"//input[@id='searchboxinput']")[0]
        search_bar.clear()
        try:
            search_bar.send_keys(search_string)
            search_bar.send_keys(Keys.ENTER)
        except:
            print("Search failed")

    def get_place_data(self, sth2vec_obj, response=None):
        if response is None:
            response = BeautifulSoup(self.driver.page_source, 'html.parser')
        place_url = self.driver.current_url
        type_of_object = self.get_place_type(response)
        name = response.find(class_=re.compile(self.html_markers_dict['place_name'])).text[1:-2]
        cluster = sth2vec_obj.classify_type_of_object(type_of_object)
        response = BeautifulSoup(self.driver.page_source, 'html.parser')
        address = response.find(class_=self.html_markers_dict['place_address'])
        localization = self.get_place_location(address.text)
        return [name, place_url, type_of_object, cluster, localization]

