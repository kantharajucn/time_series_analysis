import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.dates as mdates
from lib.constants import ARIMA_RESULTS, PREDICTIONS, PREDICTIONS_TEN_YEARS
from lib.data_preparation import DataFrameInitialisation
from statsmodels.tsa.arima_model import ARMAResults



class ArimaModel(DataFrameInitialisation):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.log_shift_data = self.get_log_shift_dataset()
        self.log_scale_data = self.get_log_scale_dataset()

    def arima_model(self):
        """
        1. Training ARIMA model using the found P, D, and Q values from the previous tests.
        2. Printing the model fitting results.
        3. Plotting prediction for 10 years, which is from 2011 to 2020.
        4. Forecasting for three years using the trained ARIMA model
        5. Printing out results

        """
        model = ARIMA(self.log_scale_data, order=(2, 1, 1))
        results = model.fit(disp=-1)
        fig, ax = plt.subplots(figsize=(15, 7))
        plt.plot(self.log_shift_data)
        ax.xaxis.set_tick_params(reset=True)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.plot(results.fittedvalues, color='red')
        plt.title('RSS: %.4f' % sum(results.fittedvalues - self.log_shift_data['Dwelling units'])**2)
        plt.savefig(ARIMA_RESULTS)

        # Model fitting results
        print('AIC: {} \nBIC: {}'.format(results.aic, results.bic))

        # Final predictions
        predictions_diff = pd.Series(results.fittedvalues, copy=True)
        predictions_diff_cumulative_sum = predictions_diff.cumsum()
        print(predictions_diff_cumulative_sum.head())

        predictions_log =  pd.Series(self.log_scale_data['Dwelling units'].iloc[0], index=self.log_scale_data.index)
        predictions_log = predictions_log.add(predictions_diff_cumulative_sum, fill_value=0)
        final_predictions = np.exp(predictions_log)
        print(final_predictions.head())

        # Prediction plot
        plt.subplots(figsize=(15, 7))
        results.plot_predict(1, 109)
        plt.title('Predictions for 10 years')
        plt.savefig(PREDICTIONS_TEN_YEARS)

        # Forecasting for three years
        forecast_results = results.forecast(steps=36)
        forecast_results = forecast_results[0]
        forecast_results_exp = np.exp(forecast_results)

        print(forecast_results_exp)

