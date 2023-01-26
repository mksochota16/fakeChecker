# **FakeChecker**
**FakeChecker** is a part of my Engineering thesis project on Warsaw University of Technology. Its a self hosted tool (in a form of FastAPI service) for predicting wheteher reveiws on Google Maps can be trusted or are suspicous. Its pretrained models that analyse text features are tuned for Polish data, however one can train diffrent models for different languages and the tool will work with them. Pretrained models for English might come out soon.

## Requirements
- python 3.10+
- libraries listed in requirements.txt
- MongoDB either self hosted or in cloud
- Access to at least one of Geocode APIs below:
  - [Google Maps geocode API](https://developers.google.com/maps/documentation/geocoding/overview)
  - [positionstack](https://positionstack.com/)
  - [Geoapify](https://www.geoapify.com/geocoding-api)
- `.env` file with at least one Geocode API key, template below:
```
POSITIONSTACK_GEOCODE_API_KEY=""
GEOAPIFY_GEOCODE_API_KEY=""
GOOGLE_GEOCODE_API_KEY=""

ADMIN_API_KEY="<SECRET OF OUR CHOICE>"

MONGODB_URI=""
MONGODB_PORT=""
MONGODB_NEW_DB_NAME=""
```
