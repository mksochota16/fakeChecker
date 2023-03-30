from enum import Enum


class AttributesModes(Enum):
    BASIC = "basic"
    SENTIMENT = "app/data/formatted_reviews_data_w_sentiment.csv"
    SENTIMENT_CAPS_INTER = "app/data/formatted_reviews_data_w_sentiment_caps_inter.csv"
    SIMPLE_NLP = "app/data/formatted_reviews_data_w_simple_nlp.csv"
    LESS_NLP = "app/data/formatted_reviews_data_w_less_nlp.csv"
    ALL_NLP = "app/data/formatted_reviews_data_w_all_nlp.csv"

    BEST = SENTIMENT_CAPS_INTER



