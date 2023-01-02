from typing import Optional, Tuple

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

from app.config import STH2VEC
from app.dao.dao_accounts_new import DAOAccountsNew
from app.dao.dao_reviews_new import DAOReviewsNew
from app.dao.dao_reviews_partial import DAOReviewsPartial
from app.models.account import AccountNew
from app.models.base_mongo_model import MongoObjectId
from app.models.review import ReviewNew, ReviewPartial
from app.models.place import Place
from app.models.types_cluster import CLUSTER_TYPES
from app.models.position import Position as PositionNew

from app.dao.dao_places import DAOPlaces

from app.services.database.database import Database
from app.services.scraper.tools.html_markers_tools import HTMLMarkers
from app.services.scraper.tools.info_scrape_tools import *
from app.services.scraper.tools import io_files_handler
from app.services.scraper.tools.simple_scrape_tools import *
from app.services.scraper.tools.info_scrape_tools import convert_from_relative_to_absolute_date


class ScraperUsage:
    def __init__(self, headless=True):
        options = Options()
        options.headless = headless
        self.driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
        self.html_markers_dict = io_files_handler.get_saved_html_markers()
        self.simple_scrape_tools = SimpleScrapeTools(self.driver, self.html_markers_dict)
        self.info_scrape_tools = InfoScrapeTools(self.driver, self.simple_scrape_tools, self.html_markers_dict)
        self.html_markers = HTMLMarkers(self.driver, self.simple_scrape_tools, self.info_scrape_tools,
                                        self.html_markers_dict)
        self.og_database = Database(original_database=True)
        self.new_database = Database(original_database=False)

    def collect_data_from_place(self, url, max_scroll_seconds: int = 5) -> Tuple[MongoObjectId,int]:
        self.simple_scrape_tools.start_scraping(url)
        self.info_scrape_tools.wait_for_place_site_to_load()

        response = BeautifulSoup(self.driver.page_source, 'html.parser')
        place_name: str = response.find(class_=self.html_markers_dict['place_name']).text.strip()
        place_address: str = response.find(class_=self.html_markers_dict['place_address']).text.strip()
        place_localization: PositionNew = geocode_api.forward_geocode(place_address, new_model=True)
        number_of_reviews: int = self.info_scrape_tools.get_number_of_reviews_of_place(response)
        place_rating: float = float(
            response.find(class_=self.html_markers_dict['place_rating']).contents[0].text.replace(',', '.'))
        place_url: str = self.driver.current_url
        type_of_object: str | None = self.info_scrape_tools.get_place_type(response)
        place_cluster: CLUSTER_TYPES = STH2VEC.classify_type_of_object(type_of_object)

        dao_places: DAOPlaces = DAOPlaces()
        place: Place = Place(
            name=place_name,
            url=place_url,
            address=place_address,
            localization=place_localization,
            rating=place_rating,
            number_of_reviews=number_of_reviews,
            cluster=place_cluster,
            type_of_object=type_of_object
        )

        place_id: MongoObjectId = dao_places.insert_one(place)

        self.simple_scrape_tools.wait_for_element_and_click(By.XPATH,
                                                            "//div[contains(@jsaction,'pane.rating.moreReviews')]")  # constant value for now
        self.simple_scrape_tools.scroll_down(max_seconds=max_scroll_seconds)

        response = BeautifulSoup(self.driver.page_source, 'html.parser')

        reviews_sections = response.find_all(class_=self.html_markers_dict['place_single_reviewer_section'])
        misread_reviews = 0
        for reviewer_section in reviews_sections:
            try:
                reviewer_name = self.info_scrape_tools.find_using_html_marker(self.html_markers_dict['place_reviewer_name'],
                                                                              reviewer_section).text.strip()
                content: str = self.info_scrape_tools.find_using_html_marker(
                    self.html_markers_dict['place_reviewer_content'], reviewer_section).text.strip()
                stars: int = int(self.info_scrape_tools.find_using_html_marker(
                    self.html_markers_dict['all_reviewer_stars'], reviewer_section).attrs['aria-label'][1])
                reviewer_url: str = \
                self.info_scrape_tools.find_using_html_marker(self.html_markers_dict['place_reviewer_url'],
                                                              reviewer_section).attrs['href']
                reviewer_id: str = reviewer_url.split('/')[5]
                review_id: str = \
                self.info_scrape_tools.find_using_html_marker(self.html_markers_dict['place_reviewer_review_id'],
                                                              reviewer_section).attrs['data-review-id']
                date_relative: str = self.info_scrape_tools.find_using_html_marker(
                    self.html_markers_dict['place_reviewer_date'],
                    reviewer_section).text.replace("Nowa", "").strip()
                date_absolute: datetime = convert_from_relative_to_absolute_date(date_relative)
                profile_picture_src: str = \
                self.info_scrape_tools.find_using_html_marker(self.html_markers_dict['place_reviewer_png'],
                                                              reviewer_section).attrs['src']

                try:
                    detailed_info = self.info_scrape_tools.find_using_html_marker(
                        self.html_markers_dict['place_reviewer_local_guide_and_reviews'],
                        reviewer_section).text.split(' ')
                    is_local_guide: bool = len(detailed_info) > 2
                    number_of_reviews: int = self.info_scrape_tools.get_number_of_reviews_from_split_text(detailed_info[3:])
                except:
                    is_local_guide: bool = False
                    number_of_reviews: int = 1

                photos_urls: List[str] = []
                try:
                    photo_section = self.info_scrape_tools.find_using_html_marker(
                        self.html_markers_dict['place_reviewer_photo_section'], reviewer_section)
                    if photo_section is not None:
                        photos = self.info_scrape_tools.find_using_html_marker(self.html_markers_dict['place_reviewer_photo'],
                                                                               photo_section, multiple=True)
                        for photo in photos:
                            photos_urls.append(photo.attrs['style'].split("\'")[1])
                except:
                    pass

                try:
                    review_response: Optional[str] = self.info_scrape_tools.find_review_response(reviewer_section)
                except:
                    review_response: Optional[str] = None

                dao_reviews:DAOReviewsNew = DAOReviewsNew()
                review: ReviewNew = ReviewNew(
                    review_id=review_id,
                    rating=stars,
                    content=content,
                    reviewer_url=reviewer_url,
                    reviewer_id=reviewer_id,
                    photos_urls=photos_urls,
                    response_content=review_response,
                    date=date_absolute,

                    is_local_guide=is_local_guide,
                    number_of_reviews=number_of_reviews,
                    profile_photo_url=profile_picture_src,
                    reviewer_name=reviewer_name,
                    place_id=place_id
                )
                dao_reviews.insert_one(review)
            except AttributeError:
                # sometimes the last reviews do not load in as the whole and only some attributes are present, so we skip them
                misread_reviews += 1

        return place_id, misread_reviews

    def collect_data_from_person(self, url: str, max_scroll_seconds: int = 5) -> Tuple[MongoObjectId,int]:
        dao_accounts_new: DAOAccountsNew = DAOAccountsNew()
        self.simple_scrape_tools.start_scraping(url)
        self.info_scrape_tools.wait_for_person_to_load()
        response = BeautifulSoup(self.driver.page_source, 'html.parser')
        reviewer_url = self.driver.current_url
        reviewer_id = reviewer_url.split('/')[5]

        name = response.find(class_=re.compile(self.html_markers_dict['reviewer_name'])).text
        local_guide_info = response.find(class_=re.compile(self.html_markers_dict['reviewer_guide_level'])).text.split()
        if len(local_guide_info) > 3:  # then he is a local guide
            local_guide_level = int(local_guide_info[4])
        else:  # he is not a local guide
            local_guide_level = 0

        if self.info_scrape_tools.is_private_account(response):
            number_of_reviews = None
            account_to_create: AccountNew = AccountNew(
                name=name,
                reviewer_url=reviewer_url,
                reviewer_id=reviewer_id,
                local_guide_level=local_guide_level,
                number_of_reviews=number_of_reviews,
                is_private=True
            )

            account_id: MongoObjectId = dao_accounts_new.insert_one(account_to_create)
            return account_id

        number_of_reviews = self.info_scrape_tools.get_number_of_reviews_of_person()  # int(number_of_reviews.text.split()[0])
        account_to_create: AccountNew = AccountNew(
            name=name,
            reviewer_url=reviewer_url,
            reviewer_id=reviewer_id,
            local_guide_level=local_guide_level,
            number_of_reviews=number_of_reviews,
            is_private=False
        )
        account_id: MongoObjectId = dao_accounts_new.insert_one(account_to_create)
        self.simple_scrape_tools.scroll_down(max_seconds=max_scroll_seconds)
        response = BeautifulSoup(self.driver.page_source, 'html.parser')
        reviews_sections = response.find_all(class_=re.compile(self.html_markers_dict['place_single_reviewer_section']))
        misread_reviews = 0
        for reviewer_section in reviews_sections:
            try:
                review_id: str = reviewer_section.attrs['data-review-id']

                place_name: str = self.info_scrape_tools.find_using_html_marker(
                    self.html_markers_dict['reviewer_review_label'], reviewer_section).text.strip()
                place_address: str = self.info_scrape_tools.find_using_html_marker(
                    self.html_markers_dict['place_reviewer_local_guide_and_reviews'], reviewer_section).text.strip()[:-2]
                place_localization: PositionNew = geocode_api.forward_geocode(place_address, new_model=True)

                rating: str = self.info_scrape_tools.find_using_html_marker(
                    self.html_markers_dict['all_reviewer_stars'], reviewer_section).attrs['aria-label'].strip().split(" ")[0]
                rating: int = int(rating)

                date_relative: str = self.info_scrape_tools.find_using_html_marker(
                    self.html_markers_dict['place_reviewer_date'],
                    reviewer_section).text.replace("Nowa", "").strip()
                date_absolute: datetime = convert_from_relative_to_absolute_date(date_relative)

                content: str = self.info_scrape_tools.find_using_html_marker(
                    self.html_markers_dict['reviewer_reviews_content'], reviewer_section).text.strip()

                try:
                    response_content_html_marker = self.html_markers_dict['place_reviewer_response_content'][2:]
                    response_content: str = self.info_scrape_tools.find_using_html_marker(
                        response_content_html_marker, reviewer_section).text.strip()
                except:
                    response_content: Optional[str] = None

                photos_urls: List[str] = []
                try:
                    photo_section = self.info_scrape_tools.find_using_html_marker(
                        self.html_markers_dict['place_reviewer_photo_section'], reviewer_section)
                    if photo_section is not None:
                        photos = self.info_scrape_tools.find_using_html_marker(self.html_markers_dict['place_reviewer_photo'],
                                                                               photo_section, multiple=True)
                        for photo in photos:
                            photos_urls.append(photo.attrs['style'].split("\'")[1])
                except:
                    pass

                dao_reviews_partial: DAOReviewsPartial = DAOReviewsPartial()
                review: ReviewPartial = ReviewPartial(
                    review_id=review_id,
                    reviewer_id=reviewer_id,
                    place_name=place_name,
                    place_address=place_address,
                    place_localization=place_localization,
                    rating=rating,
                    date=date_absolute,
                    content=content,
                    response_content=response_content,
                    photos_urls=photos_urls
                )
                dao_reviews_partial.insert_one(review)
            except AttributeError:
                # sometimes the last reviews do not load in as the whole and only some attributes are present, so we skip them
                misread_reviews += 1

        return account_id, misread_reviews

    def discover_new_markers(self):
        self.html_markers.discoverNewMarkers()
        self.html_markers_dict = io_files_handler.get_saved_html_markers()
        self.info_scrape_tools = InfoScrapeTools(self.driver, self.simple_scrape_tools, self.html_markers_dict)
        self.html_markers = HTMLMarkers(self.driver, self.simple_scrape_tools, self.info_scrape_tools,
                                        self.html_markers_dict)