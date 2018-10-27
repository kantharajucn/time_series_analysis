# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from lib.constants import ACF_PACF_PLOT
from lib.data_preparation import DataFrameInitialisation


class AcfPacfInterpretation(DataFrameInitialisation):
    """
    Finding `p` and `q` values using `ACF` and `PACF` plots.
    """

    def acf_pacf_interpretation(self):
        data_log_shift = self.get_log_shift_dataset()
        lag_acf = acf(data_log_shift, nlags=10)
        lag_pacf = pacf(data_log_shift, nlags=10, method='ols')

        plt.figure(figsize=(16,7))
        plt.subplot(121)
        plt.title('Auto correlation')
        plt.plot(lag_acf)
        plt.axhline(y=0, linestyle='--', color='grey')
        plt.axhline(y=-1.96 / np.sqrt(len(data_log_shift)), linestyle='--', color='gray')
        plt.axhline(y=1.96 / np.sqrt(len(data_log_shift)), linestyle='--', color='gray')

        plt.subplot(122)
        plt.title('Partial Auto correlation')
        plt.plot(lag_pacf)
        plt.axhline(y=0, linestyle='--', color='grey')
        plt.axhline(y=-1.96 / np.sqrt(len(data_log_shift)), linestyle='--', color='gray')
        plt.axhline(y=1.96 / np.sqrt(len(data_log_shift)), linestyle='--', color='gray')
        plt.tight_layout()
        plt.savefig(ACF_PACF_PLOT)
