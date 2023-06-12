# **FakeChecker**
**FakeChecker** is a part of my Engineering thesis project on Warsaw University of Technology. Its a self hosted tool (in a form of FastAPI service) for predicting wheteher reveiws on Google Maps can be trusted or are suspicous. Its pretrained models, that analyse text features, are tuned for Polish data, however one can train diffrent models for different languages and the tool will work with them. Pretrained models for English might come out soon.

## Requirements
- python 3.10+
- libraries listed in requirements.txt
- MongoDB either self hosted or in cloud
- Access to at least one of Geocode APIs below:
  - [Google Maps geocode API](https://developers.google.com/maps/documentation/geocoding/overview)
  - [positionstack](https://positionstack.com/)
  - [Geoapify](https://www.geoapify.com/geocoding-api)
- Downloaded Word2Vec model, Gensim word2vec_800_3_polish: 
  - [Git repository of authors](https://github.com/sdadas/polish-nlp-resources#word2vec)
  - [Direct link to the specific Word2Vec model](https://witedupl-my.sharepoint.com/personal/dadass_wit_edu_pl/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fdadass%5Fwit%5Fedu%5Fpl%2FDocuments%2FModels%2Fword2vec%2Fword2vec%5F800%5F3%2E7z&parent=%2Fpersonal%2Fdadass%5Fwit%5Fedu%5Fpl%2FDocuments%2FModels%2Fword2vec&ga=1)
- At least one Geocode API key

`.env` file template is presented below:
```
POSITIONSTACK_GEOCODE_API_KEY=""
GEOAPIFY_GEOCODE_API_KEY=""
GOOGLE_GEOCODE_API_KEY=""
# At least one API_key has to be provided

GENSIM_WORD2VEC_MODEL_PATH=""

ADMIN_API_KEY="<SECRET OF YOUR CHOICE>"

MONGODB_URI=""
MONGODB_PORT=""
MONGODB_NEW_DB_NAME=""
```
