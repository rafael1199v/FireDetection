import ee
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='config/.env')

SERVICE_ACCOUNT_GEE = os.getenv("SERVICE_ACCOUNT_GEE")
PRIVATE_KEY_FILE = os.getenv("PRIVATE_KEY_FILE")
CONFIGURATION_PATH = os.getenv("CONFIGURATION_PATH")

def googleEarhtEngineAuth():
    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT_GEE, f"{CONFIGURATION_PATH}/{PRIVATE_KEY_FILE}")
    ee.Initialize(credentials)

googleEarhtEngineAuth()