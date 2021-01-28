## This variant of the script will extrapolate. By default it extrapolates the same number of data points
## in the input data set. Set the "numberOfExtrapolations" variable manually change this. Note that the 
## interval length is assumed to be the difference between the date-times of the first and second data point. 
## Also note that Py-Earth can handle data where the interval length varies. 

import numpy
import pandas as pd
from pyearth import Earth

pathToInputData = '.\\Memory.csv'
dateTimeFormat = '%d/%m/%Y %H:%M'
pathToOutputData = '.\\output.txt'

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
    numberOfExtrapolations = X.asi8.size

    # Fit an Earth model
    model = Earth()
    model.fit(X,y)

    # Print the model
    print(model.summary())

    # Add some values to X so that we extend into the future
    intervalLength = X[1] - X[0]
    newDateTime = X[-1] + intervalLength
    projectedDateTimes = X.asi8.copy()

    for i in range(numberOfExtrapolations):
        projectedDateTimes = numpy.append(projectedDateTimes, newDateTime.asi8[0])
        newDateTime = newDateTime + intervalLength

    # Make predictions
    projectedDateTimes = projectedDateTimes.reshape(-1, 1)
    y_hat = model.predict(projectedDateTimes)
    array_to_file(y_hat, pathToOutputData)


buildModel()
