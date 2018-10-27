import unittest
import os
from lib.acf_and_pacf_interpretation import AcfPacfInterpretation
from lib.arima_model import ArimaModel
from lib.constants import RAW_DATA_FILE
from lib.data_preparation import DataPreparation, DataFrameInitialisation
from lib.download_data import download_data_from_source
from lib.stationary_test import StationarityTest


class TestApplication(unittest.TestCase):

    def test_raw_data_file(self):
        """
        Scenario: Test if downloading of the data from the given URL is success and it is saved in the file.
        """
        download_data_from_source()
        self.assertTrue(os.path.exists(RAW_DATA_FILE))
        self.assertTrue(os.path.isfile(RAW_DATA_FILE))
        self.assertTrue(os.stat(RAW_DATA_FILE).st_size != 0)

    def test_dataset(self):
        """
        Scenario: Testing if the dataset is exists and it contains all the data points(73).
        Also, all the data transformations exists.
        """
        initializer = DataFrameInitialisation()
        self.assertIsNotNone(initializer.data_frame)
        self.assertIsNotNone(initializer.log_scale_data_frame)
        self.assertIsNotNone(initializer.log_scale_shift_data_frame)
        self.assertEqual(initializer.data_frame.size, 73)
        self.assertEqual(initializer.log_scale_data_frame.size, 73)
        self.assertEqual(initializer.log_scale_data_frame.size, 73)

    def test_plots(self):
        """
        Scenario: Testing if all the steps went well and plots are saved.
        """
        pre_process = DataPreparation()
        pre_process.prepare_dataset(RAW_DATA_FILE)
        stationary_test = StationarityTest()
        stationary_test.run_stationarity_tests()
        acf_pacf = AcfPacfInterpretation()
        acf_pacf.acf_pacf_interpretation()
        model = ArimaModel()
        model.arima_model()

        all_files = os.listdir('../plots')
        self.assertEqual(len(all_files), 10)

if __name__ == '__main__':
    unittest.main()