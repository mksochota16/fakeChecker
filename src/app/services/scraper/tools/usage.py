from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

from app.dao.reviewsOldDao import ReviewsOldDao
from app.services.database.database import Database
from app.services.scraper.models.review import Review
from app.services.scraper.tools.html_markers_tools import HTMLMarkers
from app.services.scraper.tools.info_scrape_tools import *
from app.services.scraper.tools import io_files_handler
from app.services.scraper.tools.simple_scrape_tools import *
from app.services.scraper.utils.magic_scroll_formula import magic_scroll_formula


class ScraperUsage:
    def __init__(self, headless=True):
        options = Options()
        options.headless = headless
        self.driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
        self.simple_scrape_tools = SimpleScrapeTools(self.driver)
        self.html_markers_dict = io_files_handler.get_saved_html_markers()
        self.info_scrape_tools = InfoScrapeTools(self.driver, self.simple_scrape_tools, self.html_markers_dict)
        self.html_markers = HTMLMarkers(self.driver, self.simple_scrape_tools, self.info_scrape_tools,
                                        self.html_markers_dict)
        self.og_database = Database(original_database=True)
        self.new_database = Database(original_database=False)

    def collect_data_from_place(self, url):
        self.simple_scrape_tools.start_scraping(url)
        self.info_scrape_tools.wait_for_place_site_to_load()

        response = BeautifulSoup(self.driver.page_source, 'html.parser')
        place_header = response.find(class_=self.html_markers_dict['place_header'])
        place_address = response.find(class_=self.html_markers_dict['place_address'])
        number_of_reviews = self.info_scrape_tools.get_number_of_reviews_of_place(response)
        type_of_object = self.info_scrape_tools.getPlaceType(response)

        self.simple_scrape_tools.wait_for_element_and_click(By.XPATH,
                                                            "//div[contains(@jsaction,'pane.rating.moreReviews')]")  # constant value for now
        self.simple_scrape_tools.scroll_down(max_seconds=5)

        response = BeautifulSoup(self.driver.page_source, 'html.parser')

        # reviewer_names = response.find_all(class_=self.html_markers_dict['place_reviewer_name'])
        # reviewer_content = response.find_all(class_=self.html_markers_dict['place_reviewer_content'])
        # reviewer_content = response.find_all(class_=self.html_markers_dict['place_reviewer_contents'])
        reviews_sections = response.find_all(class_=self.html_markers_dict['place_single_reviewer_section'])
        for reviewer_section in reviews_sections:
            reviewer_name = self.info_scrape_tools.find_using_html_marker(self.html_markers_dict['place_reviewer_name'],
                                                                          reviewer_section).text
            content = self.info_scrape_tools.find_using_html_marker(
                self.html_markers_dict['place_reviewer_content'], reviewer_section)[0].text
            stars = self.info_scrape_tools.find_using_html_marker(
                self.html_markers_dict['all_reviewer_stars'], reviewer_section).attrs['aria-label'][1]
            reviewer_id = self.info_scrape_tools.find_using_html_marker(self.html_markers_dict['place_reviewer_url'],
                                                                         reviewer_section).attrs['href'].split('/')[5]
            review_id = self.info_scrape_tools.find_using_html_marker(self.html_markers_dict['place_reviewer_review_id'],
                                                                         reviewer_section).attrs['data-review-id']
            date = self.info_scrape_tools.find_using_html_marker(self.html_markers_dict['place_reviewer_date'],
                                                                        reviewer_section).text
            profile_picture_src = self.info_scrape_tools.find_using_html_marker(self.html_markers_dict['place_reviewer_png'],
                                                                        reviewer_section).attrs['src']

            try:
                detailed_info = self.info_scrape_tools.find_using_html_marker(self.html_markers_dict['place_reviewer_local_guide_and_reviews'],
                                                                        reviewer_section).text.split(' ')
                is_local_guide = len(detailed_info) > 2
                number_of_reviews = self.info_scrape_tools.get_number_of_reviews_from_split_text(detailed_info[3:])
            except:
                is_local_guide = False
                number_of_reviews = 1

            photos_urls = []
            try:
                photo_section = self.info_scrape_tools.find_using_html_marker(self.html_markers_dict['place_reviewer_photo_section'], reviewer_section)
                photos = self.info_scrape_tools.find_using_html_marker(self.html_markers_dict['place_reviewer_photo'], photo_section, multiple=True)
                for photo in photos:
                    photos_urls.append(photo.attrs['style'].split("\'")[1])
            except:
                pass


            try:
                review_response = reviewer_section.find(
                    class_=self.html_markers_dict['place_reviewer_response_content'])
            except:
                review_response = None
        # reviewer_stars = response.find_all(class_=self.html_markers_dict['all_reviewer_stars'])
        # reviewer_date = response.find_all(class_=self.html_markers_dict['place_reviewer_date'])
        # reviewer_urls = response.find_all(class_=self.html_markers_dict['place_reviewer_url'])

    def google_maps_place_scraper(self, url):
        self.simple_scrape_tools.start_scraping(url)
        time.sleep(2)
        place_url = self.driver.current_url
        number_of_reviews = self.info_scrape_tools.get_number_of_reviews_of_place()
        self.simple_scrape_tools.wait_for_element_and_click(By.XPATH,
                                                            "//button[contains(@jsaction,'pane.rating.moreReviews')]")  # constant value for now
        self.simple_scrape_tools.scroll_down(int(number_of_reviews / 8) + 1)
        response = BeautifulSoup(self.driver.page_source, 'html.parser')
        reviewer_names = response.find_all(class_=self.html_markers_dict['place_reviewer_name'])
        reviewer_stars = response.find_all(class_=self.html_markers_dict['place_reviewer_stars'])
        reviewer_contents = response.find_all(class_=self.html_markers_dict['place_reviewer_contents'])
        reviewer_url = response.find_all(class_=self.html_markers_dict['place_reviewer_url'])
        reviews_list = []

        for (name, rating, content, url) in zip(reviewer_names, reviewer_stars, reviewer_contents, reviewer_url):
            name = str(name.text)[1:-1]
            rating = str(rating).split()[2]
            content = content.text
            url = str(url).split()[6][6:-1]
            # reviews_list.append(Review.Review(name, rating, content, url, place_url))
        time.sleep(2)
        self.driver.quit()

    def google_maps_person_scraper(self, met_people=None, place_iterator=None, person_url=None):
        if person_url is not None:
            self.simple_scrape_tools.start_scraping(person_url)
        self.info_scrape_tools.wait_for_person_to_load()
        response = BeautifulSoup(self.driver.page_source, 'html.parser')
        reviewer_url = self.driver.current_url
        name = response.find(class_=re.compile(self.html_markers_dict['reviewer_name']))
        name = name.text
        local_guide_info = response.find(class_=re.compile(self.html_markers_dict['reviewer_name_guide_level']))
        local_guide_info = local_guide_info.text.split()
        if len(local_guide_info) > 3:  # then he is a local guide
            local_guide_info = int(local_guide_info[4])
        else:  # he is not a local guide
            local_guide_info = 0
        number_of_reviews = self.info_scrape_tools.get_number_of_reviews_of_person()  # int(number_of_reviews.text.split()[0])
        self.simple_scrape_tools.scroll_down(int(number_of_reviews / 8) + 2)
        response = BeautifulSoup(self.driver.page_source, 'html.parser')
        # reviews_content = response.find_all(class_=re.compile(self.html_markers_dict['reviewer_reviews_content']))
        # reviews_stars = response.find_all(class_=re.compile("section-review-stars"))
        # reviews_place_name = response.find_all(class_=re.compile("section-review-title-consistent-with-review-text"))
        # reviews_list = []
        if met_people is None:
            met_people = []
        if place_iterator is None:
            place_iterator = 0
        while place_iterator < number_of_reviews:
            try:
                self.simple_scrape_tools.scroll_down(int(number_of_reviews / 8) + 2)
            except TimeoutException:
                self.driver.back()
                time.sleep(2)
                self.simple_scrape_tools.scroll_down(int(number_of_reviews / 8) + 2)
            place_url = self.info_scrape_tools.navigate_to_place_site(met_people, place_iterator)
            place_iterator += 1
            # if place_iterator > len(reviews_stars):
            #     break
            # reviews_list.append(Review.Review(name, rating, content, reviewer_url, place_url))

        time.sleep(1)

    def getAccountsFromPlace(self, place_url):
        accounts_list = []
        self.simple_scrape_tools.start_scraping(place_url)
        self.info_scrape_tools.get_accounts_from_localization(accounts_list, 100)

    def getReviewsOfAccounts(self, list_to_check):
        it = 0
        for account in list_to_check:
            print(f"=== Current progress = {round(it / len(list_to_check), 2)} ===")
            it += 1
            self.simple_scrape_tools.start_scraping(account.reviewer_url)
            self.info_scrape_tools.wait_for_person_to_load()
            response = BeautifulSoup(self.driver.page_source, 'html.parser')
            if not self.info_scrape_tools.is_private_account(response):
                number_of_reviews = self.info_scrape_tools.get_number_of_reviews_of_person(response)
                place_iterator = 0
                self.simple_scrape_tools.scroll_down(int(number_of_reviews / 8) + 2)
                response = BeautifulSoup(self.driver.page_source, 'html.parser')
                while place_iterator < number_of_reviews:
                    if self.info_scrape_tools.wait_for_person_to_load() < 0:
                        self.simple_scrape_tools.navigate_back(1)
                        self.info_scrape_tools.wait_for_person_to_load()
                    try:
                        try:
                            data_review_id = response.find_all(
                                class_=re.compile(self.html_markers_dict['reviewer_review_id_child']))[
                                place_iterator].parent
                        except:
                            data_review_id = response.find_all(class_=re.compile(
                                "ODSEW-ShBeI NIyLF-haAclf gm2-body-2 ODSEW-ShBeI-d6wfac ODSEW-ShBeI-SfQLQb-QFlW2 ODSEW-ShBeI-De8GHd-KE6vqe-purZT"))[
                                place_iterator]
                            print("== Unexpected marker ==")
                        data_review_id = data_review_id.attrs['data-review-id']
                        if self.new_database.is_review_already_in(data_review_id):
                            print("Was already in database")
                            place_iterator += 1
                            continue
                        content = \
                            response.find_all(class_=re.compile(self.html_markers_dict['reviewer_reviews_content']))[
                                place_iterator].text
                        stars = self.info_scrape_tools.get_stars_count(response, place_iterator)
                        place_name = \
                            response.find_all(class_=re.compile(self.html_markers_dict['reviewer_review_label']))[
                                place_iterator].text[2:-1]
                        try:
                            position = self.info_scrape_tools.get_reviewer_place_location(response, place_iterator)
                        except IndexError:
                            position = None

                        attrs = {"class_": self.html_markers_dict['reviewer_review_photo_url'],
                                 "data-review-id": data_review_id}
                        photos_info = response.find_all(attrs)
                        photos_urls = []
                        for photo in photos_info:
                            image_url = photo.attrs['style'].split('(')[1][-1]
                            photos_urls.append(image_url)
                        review_date = self.info_scrape_tools.getAbsoluteTime(response, place_iterator)
                        try:
                            response_content = response.find_all(
                                class_=re.compile(self.html_markers_dict['reviewer_reviews_response_content'][0]))[
                                place_iterator].find(
                                class_=re.compile(self.html_markers_dict['reviewer_reviews_response_content'][1])).find(
                                class_=re.compile(self.html_markers_dict['reviewer_reviews_response_content'][2]))
                            if response_content is not None:
                                response_content = response_content.text
                        except:
                            response_content = None
                        place_data = self.info_scrape_tools.getPlaceSpecificData(place_iterator)
                        if place_data is not None:
                            place_url = place_data['place_url']
                            type_of_object = place_data['type_of_object']
                        else:
                            place_url = None
                            type_of_object = None

                        review = Review(review_id=data_review_id, place_name=place_name, rating=stars, content=content,
                                        reviewer_url=account.reviewer_url, place_url=place_url,
                                        type_of_object=type_of_object,
                                        localization=position, response_content=response_content, date=review_date)
                        self.new_database.save_review(review)
                        print("Added a review to the database")
                    except IndexError:
                        print("Iterator was out of bounds")
                    except:
                        print("There was something wrong :(")
                    place_iterator += 1

    def discover_new_markers(self):
        self.html_markers.discoverNewMarkers()
        self.html_markers_dict = io_files_handler.get_saved_html_markers()
        self.info_scrape_tools = InfoScrapeTools(self.driver, self.simple_scrape_tools, self.html_markers_dict)
        self.html_markers = HTMLMarkers(self.driver, self.simple_scrape_tools, self.info_scrape_tools,
                                        self.html_markers_dict)


if __name__ == '__main__':
    usage = ScraperUsage(headless=False)
    #usage.discover_new_markers()
    # usage.collect_data_from_place(
    #     url="https://www.google.pl/maps/place/Salon+meblowy+Black+Red+White+-+meble+Warszawa/@52.1915787,20.9598395,14z/data=!4m5!3m4!1s0x471934adc20fc469:0xeffca447261b4baa!8m2!3d52.2030371!4d20.9362185")
