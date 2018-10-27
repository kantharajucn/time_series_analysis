# -*- coding: utf-8 -*-


class DownloadDataException(Exception):

    def __init__(self, data_url):
        self.message = "Could not download the data from the provided '%s'" % data_url


class DataPreparationException(Exception):
    def __init__(self):
        self.message = "Could not prepare the dataset"
