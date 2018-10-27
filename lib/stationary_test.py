# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from lib.constants import ROLLING_STATISTICS_PNG, TREND_PNG, LOG_SCALE, TIME_SERIES_COMPONENTS, \
    LOG_SCALE_MOVING_AVERAGE, DATA_SHIFT
from lib.data_preparation import DataFrameInitialisation


class StationarityTest(DataFrameInitialisation):
    """
    1. Test stationary of the time series using Rolling Statistics and Dickey-Fuller Tests.
    2. Testing components of the time series (trend, seasonality and residual).
    3. Plotting Rolling Statistics and trend graphs.
    """

    def run_stationarity_tests(self):
        self._test_stationary_original()
        self._test_stationary_log_scale_data()
        self._test_stationary_log_scale_minus_moving_average()
        self._test_stationary_data_shift()
        self._plot_trend()
        self._plot_time_series_components()

    def _stationary_test(self, time_series, file_name):
        rolling_mean = time_series.rolling(window=12).mean()
        rolling_std = time_series.rolling(window=12).std()
        fig, ax = plt.subplots(figsize=(15,7))
        time_series.plot(x_compat=True, ax=ax, color='blue', label='original')
        plt.plot(rolling_mean, color='red', label='rolling mean')
        plt.plot(rolling_std, color='black', label='rolling std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Rolling standard deviation')
        ax.xaxis.set_tick_params(reset=False)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.savefig(file_name)

    def _dickey_fuler_test(self, time_series):
        dftest= adfuller(time_series['Dwelling units'], autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test statistic', 'p-value', 'Lags used', 'Number of observations'])
        for key, value in dftest[4].items():
            dfoutput['critical values (%s)' %key]= value
        return dfoutput

    def _test_stationary_original(self):
        data_frame = self.data_frame
        file_name = ROLLING_STATISTICS_PNG
        self._stationary_test(data_frame, file_name)
        df_output = self._dickey_fuler_test(data_frame)
        print('Dickey-Fuller test results with original time series')
        print(df_output)

    def _test_stationary_log_scale_data(self):
        file_name = LOG_SCALE
        self._stationary_test(self.log_scale_data_frame, file_name)
        df_output = self._dickey_fuler_test(self.log_scale_data_frame)
        print('Dickey-Fuller test results with Log scale time series')
        print(df_output)

    def _test_stationary_log_scale_minus_moving_average(self):
        moving_average = self.log_scale_data_frame.rolling(window=12).mean()
        data_set = self.log_scale_data_frame - moving_average
        data_set.dropna(inplace=True)
        file_name = LOG_SCALE_MOVING_AVERAGE
        self._stationary_test(data_set, file_name)
        df_output = self._dickey_fuler_test(data_set)
        print('Dickey-Fuller test results with Log scale - moving average time series')
        print(df_output)

    def _test_stationary_data_shift(self):
        file_name = DATA_SHIFT
        self._stationary_test(self.log_scale_shift_data_frame, file_name)
        df_output = self._dickey_fuler_test(self.log_scale_shift_data_frame)
        print('Dickey-Fuller test results with shifting time series')
        print(df_output)

    def _plot_trend(self):
        moving_average = self.log_scale_data_frame.rolling(window=12).mean()
        fig, ax = plt.subplots(figsize=(15, 7))
        self.log_scale_data_frame.plot(x_compat=True, ax=ax, color='blue')
        ax.xaxis.set_tick_params(reset=True)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.plot(moving_average, color='red')
        plt.legend(loc='best')
        plt.title('Trend')
        plt.savefig(TREND_PNG)

    def _plot_time_series_components(self):
        dataset_log_scale = self.get_log_scale_dataset()
        decomposition = seasonal_decompose(dataset_log_scale)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        plt.figure(1)
        plt.subplot(411)
        plt.plot(dataset_log_scale, label='original')
        plt.legend(loc='best')
        plt.subplot(412)
        plt.plot(trend, label='Trend')
        plt.legend(loc='best')
        plt.subplot(413)
        plt.plot(seasonal, label='Seasonal')
        plt.legend(loc='best')
        plt.subplot(414)
        plt.plot(residual, label='Residual')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(TIME_SERIES_COMPONENTS)












