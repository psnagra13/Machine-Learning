# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Read Dataset from file
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#  HANDLING CATEGORICAL DATA ######################################################################################
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
print (X[:, 3])
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
print (X[:, 3])

onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]


#  SPLITTING DATASET INTO TRAIN AND TEST SET ######################################################################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



#  FEATURE SCALING ######################################################################################
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


print(X_train)
print(y_train)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

print (y_test)
print (y_pred)

