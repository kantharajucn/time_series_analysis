import os
import requests
import json

from lib.constants import DATA_SOURCE_URL, RAW_DATA_FILE
from lib.exceptions import DownloadDataException


def download_data_from_source():
    """
    Downloading dataset from Australian bureau of statistics
    to forecast number of dwelling units approved in three years
    """

    if os.path.exists(RAW_DATA_FILE):
        pass
    else:
        try:
            response = requests.get(DATA_SOURCE_URL)
            assert response.status_code in (200, 304)
            json_data = response.text
            data = json.loads(json_data)
            with open(RAW_DATA_FILE, 'w') as outfile:
                json.dump(data, outfile)
        except Exception:
            raise DownloadDataException(DATA_SOURCE_URL)
