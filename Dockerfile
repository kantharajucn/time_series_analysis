FROM python:latest
RUN pip install numpy
RUN pip install scipy
RUN pip install scikit-learn
RUN pip install h5py
RUN pip install pandas
RUN pip install statsmodels
RUN pip install requests
RUN pip install matplotlib

WORKDIR /time_series_analysis/lib
CMD ["python", "/time_series_analysis/run.py"]
