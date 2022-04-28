from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA

def ARIMA_model(df, p=1, q=1, d=1):
    model = ARIMA(df, order=(p, d, q))
    results_ARIMA = model.fit(disp=-1)
    rmse = np.sqrt(mean_squared_error(df[1:], results_ARIMA.fittedvalues))
    return results_ARIMA, rmse