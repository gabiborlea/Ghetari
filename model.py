import pmdarima as pm
import pandas as pd
import pickle
import matplotlib.pyplot as plt


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
        # predictions for the test set with confidence values
        predictions, conf_vals = self.__model.predict(n_periods=len(self.__test), return_conf_int=True)
        predictions = pd.Series(predictions, index=self.__test.index)

        return predictions, conf_vals

    def plot_predictions(self, predictions, conf_vals):
        lower_bounds = pd.Series(conf_vals[:, 0], index=list(self.__test.index))
        upper_bounds = pd.Series(conf_vals[:, 1], index=list(self.__test.index))
        plt.plot(self.__series)
        plt.plot(predictions, color='darkgreen')
        plt.fill_between(lower_bounds.index,
                         lower_bounds,
                         upper_bounds,
                         color='k', alpha=.15)

        plt.title("Forecast for test values")
        plt.xlabel('Year')
        plt.ylabel('Temp (Celsius)')
        plt.legend(['True', 'Predicted'])
        plt.show()
