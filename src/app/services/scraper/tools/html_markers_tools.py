import re
import time

from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By

from app.services.scraper.tools import io_files_handler
from app.services.scraper.tools import simple_scrape_tools
from app.services.scraper.tools.info_scrape_tools import InfoScrapeTools
from app.services.scraper.tools.simple_scrape_tools import SimpleScrapeTools
from app.services.scraper.utils.magic_scroll_formula import magic_scroll_formula


class HTMLMarkers:
    def __init__(self, driver, simpleScrapeTools, infoScrapeTools, html_markers_dict):
        self.driver = driver
        self.simpleScrapeTools: SimpleScrapeTools = simpleScrapeTools
        self.infoScrapeTools: InfoScrapeTools = infoScrapeTools
        self.html_markers_dict: dict = html_markers_dict

    def findClassFromDict(self, html_marker, place_response):
        html_markers = self.html_markers_dict[html_marker]
        search_iterator = 0
        search_response = place_response.find_all(class_=re.compile(html_markers[search_iterator]))
        result = self.findClassFromDictRecursive(search_response, 1, html_markers)
        return result

    def findClassFromDictRecursive(self, previous_result, level, html_markers):
        for result in previous_result:
            search_response2 = result.find_all(class_=re.compile(html_markers[level]))
            if search_response2 is not None and len(search_response2) > 0:
                if level + 1 == len(html_markers):
                    return search_response2
                else:
                    done = self.findClassFromDictRecursive(search_response2, level + 1, html_markers)
                    if done is not None:
                        return done
        return None

    def discoverNewMarkers(self):
        self.simpleScrapeTools.start_scraping(
            "https://www.google.com/maps/contrib/118323224570864445260/reviews?hl=pl")
        try:
            self.infoScrapeTools.wait_for_person_to_load()
        except KeyError:  # if there are no known markers
            time.sleep(4)
        response = BeautifulSoup(self.driver.page_source, 'html.parser')
        try:
            number_of_reviews = self.infoScrapeTools.get_number_of_reviews_of_person(response)
        except KeyError:  # if there are no known markers
            number_of_reviews = 15
        place_iterator = 0
        self.simpleScrapeTools.scroll_down(int(number_of_reviews / 8) + 2)
        time.sleep(1)
        self.simpleScrapeTools.scroll_down(int(number_of_reviews / 8) + 2)
        response = BeautifulSoup(self.driver.page_source, 'html.parser')
        content = response.find_all()
        name_dict = {}
        for div in content:
            try:
                if div.text == 'Paweł Gryka':
                    name_dict['reviewer_name'] = simple_scrape_tools.getClassName(div)
                elif div.text.split(" przewodnik")[0] == 'Lokalny':
                    name_dict['reviewer_guide_level'] = simple_scrape_tools.getClassName(div)
                elif div.text.split()[1] == 'opinia' or div.text.split()[1] == 'opinie' or div.text.split()[
                    1] == 'opinii' or div.text.split()[2] == 'opinie' or div.text.split()[2] == 'opinii':
                    name_dict['reviewer_number_of_reviews'] = simple_scrape_tools.getClassName(div)
                elif div.text == "Fair Winds - kursy src, RYA, nawigacji morskiej":
                    name_dict['reviewer_review_label'] = simple_scrape_tools.getClassName(div)
                    self.simpleScrapeTools.scroll_down(1)
                    time.sleep(2)
                    # click on the review
                    self.simpleScrapeTools.wait_for_element_and_click(By.XPATH,
                                                                f"//div[@data-review-id='ChZDSUhNMG9nS0VJQ0FnSURpNEwya093EAE']",
                                                                      0)
                    time.sleep(2)
                    buttons = BeautifulSoup(self.driver.page_source, 'html.parser').find_all()
                    for button in buttons:
                        try:
                            if button.text == 'Szczegóły miejsca':
                                name_dict['trans_more_info_button'] = simple_scrape_tools.getClassName(button)
                                self.simpleScrapeTools.wait_for_element_and_click(By.XPATH,
                                                                            f"//button[contains(@class,'{name_dict['trans_more_info_button']}')]",
                                                                                  0)
                                time.sleep(4)
                                place_response = BeautifulSoup(self.driver.page_source, 'html.parser')
                                place_info = place_response.find_all()
                                got_parameters = False
                                for info in place_info:
                                    try:
                                        if info.text == 'Szkoła żeglarstwa' and not got_parameters:
                                            name_dict['place_description'] = [
                                                simple_scrape_tools.getClassName(info.parent.parent),
                                                simple_scrape_tools.getClassName(info.parent),
                                                simple_scrape_tools.getClassName(info)]
                                            got_parameters = True
                                        elif info.attrs['jsaction'] == 'pane.rating.moreReviews':
                                            name_dict['place_number_of_reviews'] = simple_scrape_tools.getClassName(info.parent)
                                            name_dict['place_rating'] = simple_scrape_tools.getClassName(info)
                                        elif 'Fair Winds - kursy src, RYA, nawigacji morskiej' in info.text:
                                            for sub_div in info.find_all():
                                                if sub_div.text == 'Fair Winds - kursy src, RYA, nawigacji morskiej':
                                                    name_dict['place_name'] = simple_scrape_tools.getClassName(sub_div)
                                        elif 'Puławska 12/3, 02-740 Warszawa' in info.text:
                                            name_dict['place_address'] = simple_scrape_tools.getClassName(info)
                                    except:
                                        pass
                                self.infoScrapeTools.click_on_more_reviews()
                                    #wait_for_element_and_click(By.XPATH,
                                                           #                 "//button[@jsaction='pane.rating.moreReviews']")  # constant value for now
                                time.sleep(2)
                                number_of_reviews_of_place = self.infoScrapeTools.get_number_of_reviews_of_place()
                                self.simpleScrapeTools.scroll_down(magic_scroll_formula(number_of_reviews_of_place))
                                place_response = BeautifulSoup(self.driver.page_source, 'html.parser')
                                place_reviews = place_response.find_all()
                                has_content_info = False
                                has_content_response = False
                                has_date_info = False
                                for info in place_reviews:
                                    try:
                                        if info.text == 'Paweł Gryka':
                                            name_dict['place_reviewer_name'] = simple_scrape_tools.getClassName(info)
                                            name_dict['place_reviewer_local_guide_and_reviews'] = simple_scrape_tools.getClassName(
                                                info.parent.parent.contents[3])
                                            reviewer_section = info.parent.parent.parent.parent.parent.parent.parent.parent
                                            name_dict['place_single_reviewer_section'] = simple_scrape_tools.getClassName(
                                                reviewer_section)
                                            name_dict[
                                                'scrollable_div'] = simple_scrape_tools.getClassName(
                                                reviewer_section.parent.parent)
                                            name_dict['place_reviewer_url'] = simple_scrape_tools.getClassName(reviewer_section.contents[1].contents[5].contents[1].contents[1])
                                            name_dict['place_reviewer_png'] = simple_scrape_tools.getClassName(reviewer_section.contents[1].contents[5].contents[1].contents[1].contents[1])
                                        elif info.attrs['aria-label'] == ' 5 gwiazdek ':
                                            name_dict['all_reviewer_stars'] = simple_scrape_tools.getClassName(info)
                                        elif "Pełna profeska, Anna która prowadziła ten kurs to przemiła kobieta, cierpliwa i chętna do odpowiedzi na każde pytanie. Polecam" in info.text and not has_content_info:
                                            children = info.find_all()
                                            for child in children:
                                                if child.text == "Pełna profeska, Anna która prowadziła ten kurs to przemiła kobieta, cierpliwa i chętna do odpowiedzi na każde pytanie. Polecam":
                                                    name_dict['place_reviewer_content'] = [
                                                        simple_scrape_tools.getClassName(child.parent),
                                                        simple_scrape_tools.getClassName(child)]
                                                    name_dict['place_reviewer_contents'] = simple_scrape_tools.getClassName(
                                                        child.parent)
                                                    has_content_info = True
                                                    break
                                        elif info.attrs['data-review-id'] == 'ChZDSUhNMG9nS0VJQ0FnSURpNEwya093EAE':
                                            name_dict['place_reviewer_review_id'] = simple_scrape_tools.getClassName(info)
                                        elif 'temu' in info.text and not has_date_info:
                                            children = info.find_all()
                                            for child in children:
                                                if 'temu' in child.text and len(child.text) < 25:
                                                    name_dict['place_reviewer_date'] = simple_scrape_tools.getClassName(child)
                                                    has_date_info = True
                                                    break

                                        elif 'Pozdrawiamy i dziękujemy!' in info.text and not has_content_response:
                                            children = info.find_all()
                                            for child in children:
                                                if child.text == 'Pozdrawiamy i dziękujemy!':
                                                    name_dict['place_reviewer_response_content'] = [
                                                        simple_scrape_tools.getClassName(child.parent.parent.parent),
                                                        simple_scrape_tools.getClassName(child.parent.parent),
                                                        simple_scrape_tools.getClassName(child.parent),
                                                        simple_scrape_tools.getClassName(child)]
                                                    has_content_response = True
                                                    break
                                        elif '118323224570864445260' in info.attrs['href']:
                                            name_dict['place_reviewer_reviewer_id'] = simple_scrape_tools.getClassName(info)
                                            name_dict['place_reviewer_img'] = simple_scrape_tools.getClassName(
                                                info.child)

                                    except:
                                        pass
                        except:
                            pass
                    self.driver.back()
                    time.sleep(1)
                    self.driver.back()
                    time.sleep(1)
                if 'Ładne ale ważna jest dobra pogoda' in div.text and 'reviewer_reviews_content' not in name_dict:
                    for sub_div in div.find_all():
                        if sub_div.text == 'Ładne ale ważna jest dobra pogoda':
                            name_dict['reviewer_reviews_content'] = simple_scrape_tools.getClassName(sub_div.parent)
                            name_dict['reviewer_review_id_child'] = simple_scrape_tools.getClassName(
                                sub_div.parent.parent.parent)
                            break
                if '(Przetłumaczone przez Google) Super Paweł dziękuję bardzo' in div.text and 'reviewer_reviews_response_content' not in name_dict:
                    for sub_div in div.find_all():
                        if sub_div.text == '(Przetłumaczone przez Google) Super Paweł dziękuję bardzo\n\n(Wersja oryginalna)\nSuper Pawel thankyou so much':
                            name_dict['reviewer_reviews_response_content'] = [
                                simple_scrape_tools.getClassName(sub_div.parent.parent.parent),
                                simple_scrape_tools.getClassName(sub_div.parent),
                                simple_scrape_tools.getClassName(sub_div)]
                            break
                if 'Puławska 12/3, 02-740 Warszawa' in div.text and len(div.text) < 100 and 'reviewer_review_location' not in name_dict:
                    for sub_div in div.find_all():
                        if sub_div.text == 'Puławska 12/3, 02-740 Warszawa':
                            name_dict['reviewer_review_location'] = simple_scrape_tools.getClassName(sub_div.parent)
                            break
                    name_dict['reviewer_review_location'] = simple_scrape_tools.getClassName(div)
                for number in range(2, 4):
                    if f'{number} lata temu' in div.text and 'reviewer_review_relative_date' not in name_dict:
                        for sub_div in div.find_all():
                            if sub_div.text == f'{number} lata temu':
                                name_dict['reviewer_review_relative_date'] = simple_scrape_tools.getClassName(sub_div)
                                break
                    if 'reviewer_review_relative_date' in name_dict:
                        break

            except:
                pass

        io_files_handler.save_new_html_markers(name_dict)
        self.html_markers_dict = io_files_handler.get_saved_html_markers()
        self.simpleScrapeTools = SimpleScrapeTools(self.driver)
        self.infoScrapeTools = InfoScrapeTools(self.driver, self.simpleScrapeTools, self.html_markers_dict)

        self.extract_private_account_marker(name_dict)
        self.extract_special_hotel_marker(name_dict)
        self.extract_photo_url(name_dict)
        self.extract_special_search_results_marker(name_dict)
        self.extract_place_review_photo_marker(name_dict)

        io_files_handler.save_new_html_markers(name_dict)
        self.html_markers_dict = io_files_handler.get_saved_html_markers()
        self.simpleScrapeTools = SimpleScrapeTools(self.driver)
        self.infoScrapeTools = InfoScrapeTools(self.driver, self.simpleScrapeTools, self.html_markers_dict)


    def extract_private_account_marker(self, name_dict):
        #  Private account
        self.simpleScrapeTools.start_scraping(
            "https://www.google.com/maps/contrib/111015699572765545347/reviews/@52.1915787,20.9598395,14z/data=!4m3!8m2!3m1!1e1?hl=pl-PL")
        self.infoScrapeTools.wait_for_person_to_load()
        response = BeautifulSoup(self.driver.page_source, 'html.parser')
        content = response.find_all()
        for div in content:
            if div.text == "Ten użytkownik nie napisał jeszcze żadnej opinii lub nie chce jej wyświetlać.":
                name_dict['reviewer_private_label'] = simple_scrape_tools.getClassName(div)
        io_files_handler.save_new_html_markers(name_dict)
        self.html_markers_dict = io_files_handler.get_saved_html_markers()

    def extract_special_hotel_marker(self, name_dict):
        # Hotels special markers
        self.simpleScrapeTools.start_scraping(
            "https://www.google.com/maps/place/NYX+Hotel+Warsaw/@52.228435,20.9913912,16z/data=!4m8!3m7!1s0x471ecd9cb86997ef:0xf64528434564edb5!5m2!4m1!1i2!8m2!3d52.2284355!4d20.9987316?hl=pl-PL")
        self.infoScrapeTools.wait_for_place_site_to_load()
        self.infoScrapeTools.click_on_more_reviews()
        response = BeautifulSoup(self.driver.page_source, 'html.parser')
        content = response.find_all()
        for div in content:
            if div.text == "5/5" and 'hotel_rating_label' not in name_dict:
                name_dict['hotel_rating_label'] = simple_scrape_tools.getClassName(div)
                break
        io_files_handler.save_new_html_markers(name_dict)
        self.html_markers_dict = io_files_handler.get_saved_html_markers()

    def extract_photo_url(self, name_dict):
        # Photo url marker
        self.simpleScrapeTools.start_scraping(
            "https://www.google.com/maps/contrib/118123676958528776693/reviews/@52.4530353,19.3817185,7z/data=!3m1!4b1!4m3!8m2!3m1!1e1?hl=pl-PL")
        self.infoScrapeTools.wait_for_person_to_load()
        response = BeautifulSoup(self.driver.page_source, 'html.parser')
        number_of_reviews = self.infoScrapeTools.get_number_of_reviews_of_person(response)
        self.simpleScrapeTools.scroll_down(int(number_of_reviews / 8) + 2)
        response = BeautifulSoup(self.driver.page_source, 'html.parser')
        content = response.find_all()
        for div in content:
            try:
                jsaction = div.attrs['jsaction']
                if 'pane.review.openPhoto' in jsaction and 'reviewer_review_photo_url' not in name_dict:
                    name_dict['reviewer_review_photo'] = simple_scrape_tools.getClassName(div)
                    name_dict['reviewer_review_photo_section'] = simple_scrape_tools.getClassName(div.parent)
                    break
            except KeyError:
                pass
        io_files_handler.save_new_html_markers(name_dict)
        self.html_markers_dict = io_files_handler.get_saved_html_markers()

    def extract_special_search_results_marker(self, name_dict):
        # Hotels special markers
        self.simpleScrapeTools.start_scraping("https://www.google.pl/maps/search/++iDared+SerwisSerwis+MacBook+Warszawa+%7C+Serwis+iPhone+Warszawa+%7C+Wymiana+Szybki+%7C+Baterii+%7C+Wy%C5%9Bwietlacza+%7C+iPad+%7C+Apple+%7C+Watch/@52.2261066,21.008886,13z")
        self.infoScrapeTools.wait_for_place_site_to_load()
        response = BeautifulSoup(self.driver.page_source, 'html.parser')
        content = response.find_all()
        for div in content:
            if div.text == "Naprawa telefonów komórkowych" and 'search_result_marker_0' not in name_dict:
                name_dict['search_result_marker_0'] = simple_scrape_tools.getClassName(div.parent.parent.parent.parent.parent.parent.parent.parent.parent.parent.parent)
                break

        self.simpleScrapeTools.start_scraping(
            "https://www.google.pl/maps/search/Restauracja+STATEK/@52.1916469,20.9074379,12z/data=!3m1!4b1")

        self.infoScrapeTools.wait_for_place_site_to_load()
        self.simpleScrapeTools.scroll_down(5)
        response = BeautifulSoup(self.driver.page_source, 'html.parser')
        content = response.find_all()
        for div in content:
            if div.text == "Restauracja" and 'search_result_marker_1' not in name_dict:
                name_dict['search_result_marker_1'] = simple_scrape_tools.getClassName(
                    div.parent.parent.parent.parent.parent.parent.parent.parent.parent.parent.parent)
                break

        io_files_handler.save_new_html_markers(name_dict)
        self.html_markers_dict = io_files_handler.get_saved_html_markers()


    def extract_place_review_photo_marker(self, name_dict):
        # Place review photo marker
        self.simpleScrapeTools.start_scraping("https://www.google.pl/maps/place/Park+Szcz%C4%99%C5%9Bliwicki/@52.210643,20.9902988,13z/data=!4m7!3m6!1s0x471eccaa57d8b4d3:0x51864b2fd0100544!8m2!3d52.2047413!4d20.9593258!9m1!1b1")
        self.infoScrapeTools.wait_for_place_site_to_load()
        self.simpleScrapeTools.scroll_down(max_seconds=5)
        response = BeautifulSoup(self.driver.page_source, 'html.parser')
        reviews_sections = response.find_all(class_=self.html_markers_dict['place_single_reviewer_section'])
        marker_found_flag = False
        for reviews_section in reviews_sections:
            content = reviews_section.find_all()
            for div in content:
                try:
                    jsaction = div.attrs['jsaction']
                    if 'openPhoto' in jsaction:
                        name_dict['place_reviewer_photo'] = simple_scrape_tools.getClassName(div)
                        name_dict['place_reviewer_photo_section'] = simple_scrape_tools.getClassName(div.parent)
                        marker_found_flag = True
                        break
                except KeyError:
                    continue
            if marker_found_flag:
                break
        io_files_handler.save_new_html_markers(name_dict)
        self.html_markers_dict = io_files_handler.get_saved_html_markers()
