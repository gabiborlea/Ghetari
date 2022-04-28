import csv
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import pmdarima as pm
import pickle

file = open("resources/Environment_Temperature_change_E_All_Data_NOFLAG.csv", encoding = "ISO-8859-1")

csvreader = csv.reader(file)
header = []
header = next(csvreader)

rows = []
for row in csvreader:
    rows.append(row)

# extract month names
month_names = np.array(rows[:24])[::2, 3]
# extract numerical values for each year
antartica = np.array(rows[238:262])[::2, 7:].astype(float)

# generate additional data for plots
years = np.arange(1961, 2020, 1)
month_idx = np.arange(1, len(antartica) + 1, 1)

# print(afghanistan)

temperatures = []
dates = []
month = 1

for j in range(len(antartica[0])):
    for i in range(len(antartica)):
        year = 1961 + j
        month = i + 1
        temperatures.append(antartica[i][j])
        # print(month, year, antartica[i][j])
        dates.append(datetime.datetime(year, month, 1))


series = pd.DataFrame(temperatures)
series.index = dates

train = series[:-int(len(series)/10)]
test = series[-int(len(series)/10):]
# print(series)
# print(train)
# print(test)
#
model = pm.auto_arima(train.values,
                      start_p=1, # lag order starting value
                      start_q=1, # moving average order starting value
                      test='adf', #ADF test to decide the differencing order
                      max_p=3, # maximum lag order
                      max_q=3, # maximum moving average order
                      m=12, # seasonal frequency
                      d=None, # None so that the algorithm can chose the differencing order depending on the test
                      seasonal=True,
                      start_P=0,
                      D=1, # enforcing the seasonal frequencing with a positive seasonal difference value
                      trace=True,
                      suppress_warnings=True,
                      stepwise=True)

print(model.summary())

# save the model
with open('seasonal_o.pkl', 'wb') as f:
    pickle.dump(model, f)

# predictions for the entire series
all_vals = model.predict(n_periods=len(series), return_conf_int=False)
all_vals = pd.Series(all_vals, index=series.index)
plt.plot(series)
plt.plot(all_vals, color='darkgreen')
plt.title('Forecast values for the entire series')
plt.xlabel('Year')
plt.ylabel('Temp (Celcius)')
plt.legend(['True', 'Predicted'])
plt.show()

# predictions for the test set with confidence values
preds, conf_vals = model.predict(n_periods=len(test), return_conf_int=True)
preds = pd.Series(preds, index=test.index)

lower_bounds = pd.Series(conf_vals[:, 0], index=list(test.index))
upper_bounds = pd.Series(conf_vals[:, 1], index=list(test.index))

plt.plot(series)

plt.plot(preds, color='darkgreen')
plt.fill_between(lower_bounds.index,
                 lower_bounds,
                 upper_bounds,
                 color='k', alpha=.15)

plt.title("Forecast for test values")
plt.xlabel('Year')
plt.ylabel('Temp (Celcius)')
plt.legend(['True', 'Predicted'])
plt.show()


# plot data for each month
# for i in range(len(afghanistan)):
#     plt.plot(years, afghanistan[i], marker='o')
#     plt.title("Afghanistan - temperature change for month " + month_names[i])
#     plt.yticks([])
#     plt.show()

# # average data to have one value for each month
# avg_month = np.average(list(afghanistan), axis=1)
# plt.plot(month_idx, avg_month, marker='o')
# plt.title("Average temperature change for months")
# plt.xticks(month_idx)
# plt.show()
#
# avg = np.average(list(afghanistan), axis=0)
# plt.plot(years, avg, marker='o')
# plt.title("Average temperature change for years")
# plt.show()

