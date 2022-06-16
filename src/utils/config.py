"""Config values"""
from os import environ, path
from dotenv import load_dotenv

basedir = path.abspath(path.dirname(__file__))
load_dotenv(path.join(basedir, '.env'))


# Data cleaning
index = environ.get('INDEX_DF')
orig_data_path = environ.get('ORIG_DATA_PATH')
cleaned_data_path = environ.get('CLEANED_DATA_PATH')
value_counts_path = environ.get('VALUE_COUNTS_PATH')

# Model
target = environ.get('TARGET')