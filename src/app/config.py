import os
from dotenv import load_dotenv

load_dotenv()
POSITIONSTACK_API_KEY = os.getenv("POSITIONSTACK_GEOCODE_API_KEY")