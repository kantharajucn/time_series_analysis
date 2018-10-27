# -*- coding: utf-8 -*-

import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from lib.constants import DATA_FRAME_CSV, PATTERN, ORIGINAL_TIME_SERIES
from lib.exceptions import DataPreparationException


class DataPreparation():
    """
    Preparing final time series dataset from the downloaded json data.
    """

    def _read_data(self, file_name):
        with open(file_name) as json_data:
            data = json.loads(json_data.read())
            return data

    def _extract_observations(self, data):
        """
        Extracting only the observations which are located at index `0`.
        Index `1` contains the meta data of the observations which are not required for the task.
        """
        observations = data['dataSets'][0]['observations']
        return observations

    def _extract_time_series(self, observations):
        time_series_dict = dict()
        pattern = re.compile(PATTERN)
        for key, values in observations.items():
            if pattern.match(key) is not None:
                month = key.split(':')[-1]
                time_series_dict[month] = values[0]
        return time_series_dict

    def prepare_dataset(self, data_file):
        try:
            json_data = self._read_data(data_file)
            observations = self._extract_observations(json_data)
            single_time_series = self._extract_time_series(observations)
            df = pd.DataFrame(list(single_time_series.items()), columns=['months', 'Dwelling units'])
            time_stamp_index = pd.date_range(start='2011-07-01', periods=73, freq='M').tolist()
            df['months'] = time_stamp_index
            df.set_index('months', inplace=True)
            df.to_csv(DATA_FRAME_CSV)
        except:
            raise DataPreparationException()


class DataFrameInitialisation:
    """
    Read dataset from file and initialise the dataset with different transformations.
    """

    def __init__(self):
        self.data_frame = self.get_data_frame()
        self.log_scale_data_frame = self.get_log_scale_dataset()
        self.log_scale_shift_data_frame = self.get_log_shift_dataset()

    def get_data_frame(self):
        try:
            df = pd.read_csv(DATA_FRAME_CSV)
            df['months'] = pd.to_datetime((df['months']))
            df.set_index('months', inplace=True)
            return df
        except FileNotFoundError:
            print('Provided file does not exists')

    def get_log_shift_dataset(self):
        dataset_log_scale = np.log(self.data_frame)
        dataset_log_scale.dropna(inplace=True)
        data_log_shift = dataset_log_scale - dataset_log_scale.shift()
        data_log_shift.dropna(inplace=True)
        return data_log_shift

    def get_log_scale_dataset(self):
        dataset_log_scale = np.log(self.data_frame)
        dataset_log_scale.dropna(inplace=True)
        return dataset_log_scale

    def plot_time_series(self):
        ig, ax = plt.subplots(figsize=(15, 7))
        self.data_frame.plot(x_compat=True, ax=ax, color='blue')
        plt.legend(loc='best')
        plt.title('Original time series')
        ax.xaxis.set_tick_params(reset=False)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.savefig(ORIGINAL_TIME_SERIES)
