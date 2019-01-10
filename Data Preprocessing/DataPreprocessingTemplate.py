# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Read Dataset from file
# file has 4 coloumns (0,1,2,3)
# X = 0,1,2
# Y = 3
dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:, :-1].values
# X = dataset.iloc[:, [0,1,2]].values

y = dataset.iloc[:, 3].values

#  HANDLING MISSING DATA ######################################################################################################################

from sklearn.impute import SimpleImputer
'''
    SimpleImputer(missing_values = , stratergy = , fill_value = , verbose=0, )

            missing_values = The placeholder for the missing values. All occurrences of missing_values will be imputed.
            stratergy =  mean, median, most_frequent, constant
            fill_value =  When strategy == constant, fill_value is used to replace all occurrences of missing_values. 
'''
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# handling missing data for coloumn = 1
imputer = imputer.fit(X[:, [1]])
X[:, [1]] = imputer.transform(X[:, [1]])

# handling missing data for coloumns = 1,2 together
imputer3 = SimpleImputer(missing_values=np.nan, strategy='median')
imputer3 = imputer3.fit(X[:, 1:3])
X[:, 1:3] = imputer3.transform(X[:, 1:3])

print (X)

#  HANDLING CATEGORICAL DATA ######################################################################################
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
print (X[:, 0])
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
print (X[:, 0])

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]


# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)





#  SPLITTING DATASET INTO TRAIN AND TEST SET ######################################################################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



#  FEATURE SCALING ######################################################################################
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

print(X_train)
print(y_train)