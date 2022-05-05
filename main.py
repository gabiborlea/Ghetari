from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from train_data import TrainData
from model import Model

if __name__ == '__main__':
    train_data = TrainData("resources/Environment_Temperature_change_E_All_Data_NOFLAG.csv")
    model = Model(train_data.get_temperatures(), train_data.get_dates())
    predictions, confidences = model.predict_test_set()
    model.plot_predictions(predictions, confidences)
    predictions = model.predict_year(2022)
    model.plot_predictions(predictions, confidences)
