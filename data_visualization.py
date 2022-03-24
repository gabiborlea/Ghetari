import csv
import numpy as np
import matplotlib.pyplot as plt

file = open("resources/Environment_Temperature_change_E_All_Data_NOFLAG.csv")
csvreader = csv.reader(file)
header = []
header = next(csvreader)

rows = []
for row in csvreader:
    rows.append(row)

# extract month names
month_names = np.array(rows[:34])[::2, 3]
# extract numerical values for each year
afghanistan = np.array(rows[:34])[::2, 7:].astype(float)

# generate additional data for plots
years = np.arange(1961, 2020, 1)
month_idx = np.arange(1, len(afghanistan)+1, 1)

# plot data for each month
for i in range(len(afghanistan)):
    plt.plot(years, afghanistan[i], marker='o')
    plt.title("Afghanistan - temperature change for month " + month_names[i])
    plt.yticks([])
    plt.show()

# average data to have one value for each month
avg_month = np.average(list(afghanistan), axis=1)
plt.plot(month_idx, avg_month, marker='o')
plt.title("Average temperature change for months")
plt.xticks(month_idx)
plt.show()

