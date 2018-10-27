# -*- coding: utf-8 -*-

from lib.acf_and_pacf_interpretation import AcfPacfInterpretation
from lib.arima_model import ArimaModel
from lib.constants import RAW_DATA_FILE
from lib.data_preparation import DataPreparation, DataFrameInitialisation
from lib.download_data import download_data_from_source
from lib.exceptions import DownloadDataException, DataPreparationException
from lib.stationary_test import StationarityTest


def main():
    """
    1. Downloading the data from the given data source.
    2. Preparation of the dataset.
    3. Initialisation of the Time series data-frame.
    4. Stationarity test.
    5. Finding `p` and `q` values using ACF and PACF plots.
    6. Building ARIMA model.
    7. Displaying the results.

    """
    try:
        download_data_from_source()
        preprocessor = DataPreparation()
        preprocessor.prepare_dataset(RAW_DATA_FILE)
    except DownloadDataException:
        print('Failed downloading data source...foobar')
    except DataPreparationException:
        print('Preparation of dataset failed.')

    data_initialisation = DataFrameInitialisation()
    data_initialisation.plot_time_series()

    stationary_test = StationarityTest()
    stationary_test.run_stationarity_tests()

    acf_pacf = AcfPacfInterpretation()
    acf_pacf.acf_pacf_interpretation()

    model = ArimaModel()
    model.arima_model()


if __name__ == '__main__':
    main()
