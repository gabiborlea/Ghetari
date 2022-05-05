from train_data import TrainData
from model import Model

if __name__ == '__main__':
    train_data = TrainData("resources/Environment_Temperature_change_E_All_Data_NOFLAG.csv")
    model = Model(train_data.get_temperatures(), train_data.get_dates())

    # test dataset
    predictions_test, confidences_test = model.predict_test_set()
    model.plot_dataset_and_predictions(predictions_test, confidences_test)
    model.plot_test_dataset(predictions_test)

    # evaluation metrics for the test dataset
    print("Symmetric Mean Absolute Percentage Error=", model.compute_SMAPE(model.get_test_values(), predictions_test))
    print("Root Mean Squared Error=", model.compute_RMSE(model.get_test_values(), predictions_test))
    print("Mean Squared Error=", model.compute_MSE(model.get_test_values(), predictions_test))
    print("Mean Absolute Error=", model.compute_MAE(model.get_test_values(), predictions_test))

    # given years
    predictions_until_2022 = model.predict_until_year(2022)
    model.plot_predictions(predictions_until_2022)
