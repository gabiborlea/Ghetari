import datetime
from math import sqrt
import pmdarima as pm
import pmdarima.metrics as m
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


class Model:
    def __init__(self, temperatures, dates):
        self.__model = None
        self.__train = None
        self.__test = None
        self.__series = None

        self.__split_dataset(temperatures, dates)
        self.__create_model()

    def __split_dataset(self, temperatures, dates):
        self.__series = pd.DataFrame(temperatures)
        self.__series.index = dates

        self.__train = self.__series[:-int(len(self.__series) / 10)]
        self.__test = self.__series[-int(len(self.__series) / 10):]

    def __create_model(self):
        self.__model = pm.auto_arima(self.__train.values,
                                     start_p=1,  # lag order starting value
                                     start_q=1,  # moving average order starting value
                                     test='adf',  # ADF test to decide the differencing order
                                     max_p=3,  # maximum lag order
                                     max_q=3,  # maximum moving average order
                                     m=12,  # seasonal frequency
                                     d=None,
                                     # None so that the algorithm can chose the differencing order depending on the test
                                     seasonal=True,
                                     start_P=0,
                                     D=1,
                                     # enforcing the seasonal frequencing with a positive seasonal difference value
                                     trace=True,
                                     suppress_warnings=True,
                                     stepwise=True)
        print(self.__model.summary)

        # save the model
        with open('seasonal_o.pkl', 'wb') as f:
            pickle.dump(self.__model, f)

    def predict_all_dataset(self):
        # predictions for the entire series
        predictions = self.__model.predict(n_periods=len(self.__series), return_conf_int=False)
        predictions = pd.Series(predictions, index=self.__series.index)

        plt.plot(self.__series)
        plt.plot(predictions, color='darkgreen')
        plt.title('Forecast values for the entire series')
        plt.xlabel('Year')
        plt.ylabel('Temp (Celsius)')
        plt.legend(['True', 'Predicted'])
        plt.show()

    def predict_test_set(self):
        """
        Predict the test set with confidence values
        :return: predictions, confidence intervals
        """
        predictions, conf_vals = self.__model.predict(n_periods=len(self.__test), return_conf_int=True)
        predictions = pd.Series(predictions, index=self.__test.index)

        return predictions, conf_vals

    def predict_until_year(self, year):
        """
        Predict from 2019 until the given year
        :param year: max year to predict
        :return: predictions for the given interval
        """
        length = year - 2019 + 1
        dates = []
        for j in range(1, length):
            for i in range(1, 13):
                year = 2019 + j
                month = i
                dates.append(datetime.datetime(year, month, 1))
        predictions = self.__model.predict(n_periods=len(dates), return_conf_int=False)
        predictions = pd.Series(predictions, index=dates)

        return predictions

    def plot_dataset_and_predictions(self, predictions, conf_vals):
        """
        Plot both the dataset values (train and test) and the predictions for the test values also with confidence intervals.
        :param predictions: prediction values for the test dataset
        :param conf_vals: confidence intervals for the test dataset
        """
        lower_bounds = pd.Series(conf_vals[:, 0], index=list(self.__test.index))
        upper_bounds = pd.Series(conf_vals[:, 1], index=list(self.__test.index))
        plt.plot(self.__series)
        plt.plot(predictions, color='darkgreen')
        plt.fill_between(lower_bounds.index,
                         lower_bounds,
                         upper_bounds,
                         color='k', alpha=.15)

        plt.title("Dataset and predictions")
        plt.xlabel('Year')
        plt.ylabel('Temp (Celsius)')
        plt.legend(['True', 'Predicted'])
        plt.show()

    def plot_test_dataset(self, predictions):
        """
        Plot the predictions for the test values and the actual test values.
        :param predictions: prediction values for the test dataset
        """
        plt.plot(self.__test)
        plt.plot(predictions, color='darkgreen')
        plt.title("Test dataset (actual and predicted)")
        plt.xlabel('Year')
        plt.ylabel('Temp (Celsius)')
        plt.legend(['True', 'Predicted'])
        plt.show()

    def get_test_values(self):
        return self.__test

    @staticmethod
    def plot_predictions(predictions):
        """
        Plot given predictions
        :param predictions: prediction values
        """
        plt.plot(predictions, color='darkgreen')
        plt.title("Forecast for given years")
        plt.xticks(rotation=30)
        plt.xlabel('Year')
        plt.ylabel('Temp (Celsius)')
        plt.legend(['Predicted'])
        plt.grid()
        plt.show()

    @staticmethod
    def compute_SMAPE(actual, predicted):
        """
        Compute Symmetric Mean Absolute Percentage Error.
        A perfect SMAPE score is 0.0, and a higher score indicates a higher error rate.
        :param actual: actual values
        :param predicted: predicted values
        :return: error
        """
        return m.smape(actual, predicted)

    @staticmethod
    def compute_RMSE(actual, predicted):
        """
        Compute root mean squared error. The best value is 0.0.
        :param actual: actual values
        :param predicted: predicted values
        :return: error
        """
        return mean_squared_error(actual, predicted, squared=False)

    @staticmethod
    def compute_MSE(actual, predicted):
        """
        Compute mean squared error. The best value is 0.0.
        :param actual: actual values
        :param predicted: predicted values
        :return: error
        """
        return mean_squared_error(actual, predicted)

    @staticmethod
    def compute_MAE(actual, predicted):
        """
        Compute mean absolute error. The best value is 0.0.
        :param actual: actual values
        :param predicted: predicted values
        :return: error
        """
        return mean_absolute_error(actual, predicted)


