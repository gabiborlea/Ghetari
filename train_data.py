import csv
import numpy as np
import datetime


class TrainData:
    def __init__(self, file_path):
        self.__file = open(file_path, encoding="ISO-8859-1")
        self.__csv_reader = csv.reader(self.__file)
        self.__dates = []
        self.__temperatures = []
        self.__read_file()

    def __read_file(self):
        # discard table header
        next(self.__csv_reader)

        # save temperatures for each month
        rows = []
        for row in self.__csv_reader:
            rows.append(row)

        # extract numerical values for each year
        antarctica = np.array(rows[238:262])[::2, 7:].astype(float)
        for j in range(len(antarctica[0])):
            for i in range(len(antarctica)):
                year = 1961 + j
                month = i + 1
                self.__temperatures.append(antarctica[i][j])
                self.__dates.append(datetime.datetime(year, month, 1))

    def get_temperatures(self):
        return self.__temperatures

    def get_dates(self):
        return self.__dates

