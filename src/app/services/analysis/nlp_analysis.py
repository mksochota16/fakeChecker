# coding=ISO-8859-2
import re
from typing import List

import spacy
import stylo_metrix
from pydantic import BaseModel

from app.services.analysis.sentiment import SentimentAnalyzer
from app.services.scraper.tools.io_files_handler import import_dataframes_from_names_files
from app.services.scraper.tools.simple_scrape_tools import deEmojify


def name_has_emojis(name):
    return name != deEmojify(name)


class StyloMetrixResults(BaseModel):
    G_V: float
    G_N: float
    G_ADJ: float
    G_ADV: float
    G_PRO: float
    IN_V_INF: float
    IN_V_INFL: float
    IN_V_1S: float
    IN_V_1P: float
    IN_ADJ_POS: float
    IN_ADJ_COM: float
    IN_ADJ_SUP: float
    IN_ADV_POS: float
    IN_ADV_COM: float
    IN_ADV_SUP: float
    L_NAME: float
    L_PERSN: float
    L_PLACEN: float
    L_TCCT1: float
    L_TCCT5: float
    L_SYL_G3: float
    PS_M_VALa: float
    PS_M_AROa: float
    PS_M_DOMa: float
    PS_M_VALb: float
    PS_M_AROb: float
    PS_M_DOMb: float

    @classmethod
    def get_list_of_metrics(cls) -> List[str]:
        return ["G_V", "G_N", "G_ADJ", "G_ADV", "G_PRO", "IN_V_INF", "IN_V_INFL", "IN_V_1S", "IN_V_1P", "IN_ADJ_POS",
                   "IN_ADJ_COM", "IN_ADJ_SUP", "IN_ADV_POS", "IN_ADV_COM", "IN_ADV_SUP", "L_NAME", "L_PERSN",
                   "L_PLACEN", "L_TCCT1", "L_TCCT5", "L_SYL_G3", "PS_M_VALa", "PS_M_AROa", "PS_M_DOMa", "PS_M_VALb",
                   "PS_M_AROb", "PS_M_DOMb"]


class NLPanalysis:
    def __init__(self):
        self.name_files = import_dataframes_from_names_files()
        self.sentiment_analyzer: SentimentAnalyzer = SentimentAnalyzer("reviews_sentiment.pth")
        self.stylo_metrix = None # spacy.load('pl_nask_large')
        if self.stylo_metrix is not None:
            self.stylo_metrix.add_pipe("stylo_metrix")

    @classmethod
    def get_capslock_score(cls, text: str) -> float:
        if len(text) == 0:
            return 0
        number_capitalized = len(re.findall(r'[A-Z]', text))
        return number_capitalized * 1000 / len(text)

    @classmethod
    def get_interpunction_score(cls, text: str) -> float:
        if len(text) == 0:
            return 0
        text = text.replace("!", ";").replace(".", ";"). \
            replace(",", ";").replace("?", ";").replace("(", ";").replace(")", ";") \
            .replace("-", ";").replace(":", ";").replace(";", ";")
        number_interpunction = len(re.findall(r';', text))
        return number_interpunction * 1000 / len(text)

    @classmethod
    def get_emotional_interpunction_score(cls, text: str) -> float:
        if len(text) == 0:
            return 0
        text = text.replace(";", "").replace("!", ";").replace("?", ";")
        number_emotional_interpunction = len(re.findall(r';', text))
        return number_emotional_interpunction * 1000 / len(text)

    @classmethod
    def get_consecutive_emotional_interpunction_score(cls, text: str) -> float:
        if len(text) == 0:
            return 0
        text = text.replace(";", "").replace("!", ";").replace("?", ";")
        number_double_consecutive_interpunction = len(re.findall(r';{2,}', text))
        number_three_consecutive_interpunction = len(re.findall(r';{3,}', text))
        return (number_double_consecutive_interpunction * 1000 + number_three_consecutive_interpunction * 2000) / len(
            text)

    @classmethod
    def get_emojis_score(cls, text: str) -> float:
        if len(text) == 0:
            return 0
        return len(re.findall(r'[\U0001F600-\U0001F64F]', text)) * 1000 / len(text)

    def get_stylo_metrix_metrics(self, text: str) -> StyloMetrixResults:
        if self.stylo_metrix is None:
            raise Exception("StyloMetrix not loaded")
        doc = self.stylo_metrix(text)
        res = doc._.stylo_metrix_vector._dicts
        list_of_metrics = StyloMetrixResults.get_list_of_metrics()
        result_dict = {}
        for res_dict in res:
            if res_dict['code'] in list_of_metrics:
                result_dict[res_dict['code']] = res_dict['value']

        return StyloMetrixResults(**result_dict)

    def analyze_name_of_account(self, account_name):
        names = self.name_files[0]
        surnames = self.name_files[1]
        try:
            try:
                first_part = account_name.split(" ")[0].upper()
                second_part = account_name.split(" ")[1].upper()

                name_val_prob = names.loc[names['IMIĘ_PIERWSZE'] == first_part]
                if not name_val_prob.empty:
                    name_val = int(name_val_prob['sum'].iloc[0])
                else:
                    name_val = 0
                name_val_prob = names.loc[names['IMIĘ_PIERWSZE'] == second_part]
                if not name_val_prob.empty and name_val < int(name_val_prob['sum'].iloc[0]):
                    name_val = int(name_val_prob['sum'].iloc[0])

                surname_val_prob = surnames.loc[surnames['Nazwisko aktualne'] == first_part]
                if not surname_val_prob.empty:
                    surname_val = int(surname_val_prob['sum'].iloc[0])
                else:
                    surname_val = 0
                surname_val_prob = surnames.loc[surnames['Nazwisko aktualne'] == second_part]
                if not surname_val_prob.empty and surname_val < int(surname_val_prob['sum'].iloc[0]):
                    surname_val = int(surname_val_prob['sum'].iloc[0])

                return name_val + surname_val
            except:
                first_part = account_name.split(" ")[0].upper()
                name_val_prob = names.loc[names['IMIĘ_PIERWSZE'] == first_part]
                if not name_val_prob.empty:
                    name_val = int(name_val_prob['sum'].iloc[0])
                else:
                    name_val = 0

                surname_val_prob = surnames.loc[surnames['Nazwisko aktualne'] == first_part]
                if not surname_val_prob.empty and name_val < int(surname_val_prob['sum'].iloc[0]):
                    name_val = int(surname_val_prob['sum'].iloc[0])

                return name_val
        except:
            pass


if __name__ == '__main__':
    nlp = NLPanalysis()
