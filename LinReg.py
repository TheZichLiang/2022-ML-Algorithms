'''
Libraries imported
'''
import numpy as np
from scipy.stats import norm
import re
import datetime
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Processing Step
# I wanted to practice some regex to extract info frm IbTRACS data
myfiledata = "C:\\Users\\fzlce\\Downloads\\InternshipDataNeed\\2018-2021CombHurrReduceForMLDataWONAN-999.txt"
# load the data,get rid of the bar character, delete the last line, and split each string
data = np.loadtxt(myfiledata, delimiter=',', skiprows = 1, dtype=str)

# define input and outputs
# We want to get rid of N, E, and W and add negatives if necessary
data = np.array([item for item in data if "NaN" not in item])
for i in range(len(data)):
    data[:, 5][i] = re.sub('N', '', data[:, 5][i])
    if data[:,6][i].find('W') != -1:
        data[:, 6][i]= "-" + re.sub('W', '', data[:, 6][i])
    else:
        data[:, 6][i] = re.sub('E', '', data[:, 6][i])
INPUTS = np.delete(data, [0,1,2,7,8,9], 1)
OUTPUTS = np.array(data[:, 7:9])
print(INPUTS)
print(OUTPUTS)

# Turn the dates into Jovian format and the times /100 as ints
Date = INPUTS[:,0]
fmt = '%Y%m%d'
datearr = []
for s in Date:
    dt = datetime.datetime.strptime(s, fmt)
    tt = dt.timetuple()
    datearr.append(tt.tm_yday)
INPUTS[:,0] = datearr
print(INPUTS[:,0])
Time = INPUTS[:,1]
INPUTS[:,1] = (Time.astype(int)/100).astype(int)
print(INPUTS[:,1])

# We'll use 3 inputs first. Lat, Lon, WCStrength and our output being wind intensity. Let's convert into floats first
# Exp1Inputs = INPUTS[:, 2:5].astype(np.float64)
# Exp1Outputs = OUTPUTS[:, 0].astype(np.int64)
Exp1Inputs = INPUTS.astype(np.float64)
Exp1Outputs = OUTPUTS.astype(np.int64)

# Split into training/testing data. 
# We will use 80% as training data and 20% as testing data, so the testing data is just over 100 pieces of data
# Also let's reshaping set train
trainX, testX, trainY, testY = train_test_split(Exp1Inputs, Exp1Outputs, test_size = 0.2, random_state =0)
print(trainX.shape, trainY.shape)
trainX, trainY = shuffle(trainX, trainY)

# Let's feed this into our model print the cofficients, intercept,
linregmod = linear_model.LinearRegression()
linregmod.fit(trainX, trainY)
print("a value: ", linregmod.coef_,  " b value: ", linregmod.intercept_)

# Let's use the testing data to predict the model, then print MSE and accuracy for both speed and pressure
linregpred = linregmod.predict(testX)
diffSpeed = np.sum(np.absolute(testY[:,0] - linregpred[:,0]))/len(testY)
diffPressure = np.sum(np.absolute(testY[:,1] - linregpred[:,1]))/len(testY)
CCS, CCP = np.corrcoef(testY[:,0], linregpred[:,0])[0,1] , np.corrcoef(testY[:,1], linregpred[:,1])[0,1] 
MSES, MSEP = round(mean_squared_error(testY[:,0], linregpred[:,0]), 2), round(mean_squared_error(testY[:,1], linregpred[:,1]),2)
MAES, MAEP =  round(mean_absolute_error(testY[:,0], linregpred[:,0]), 2), round(mean_absolute_error(testY[:,1], linregpred[:,1]), 2)
print("Avg Difference: ", [diffSpeed, diffPressure])
print("Correlation Coefficient (R): ", [CCS, CCP])
print("MSE: ", [MSES, MSEP])
print("MAE: ", [MAES, MAEP])