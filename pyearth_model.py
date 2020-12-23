import numpy
import pyearth
import pandas as pd
from pyearth import Earth

pathToInputData = 'C:\\__DEMO1\\Memory.csv'
dateTimeFormat = '%d/%m/%Y %H:%M'
pathToOutputData = 'C:\\__DEMO1\\output.txt'

# Write array to file
def array_to_file(the_array, file_name):
    the_file = open(file_name, 'w')

    for item in the_array:
        the_file.write('%s\n' % item)


def buildModel():
    # Read our data
    data = pd.read_csv(pathToInputData,index_col=0)
    data.head()
    data.index = pd.to_datetime(data.index, format=dateTimeFormat)
    X = data.index._values
    X = X.reshape(-1, 1)
    y = data.values
    y = y.reshape(-1, 1)

    # Fit an Earth model
    model = Earth()
    model.fit(X,y)

    # Print the model
    print(model.summary())

    # Make predictions
    y_hat = model.predict(X)
    array_to_file(y_hat, pathToOutputData)


buildModel()
