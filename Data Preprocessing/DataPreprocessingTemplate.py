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

cols = dataset.columns
print ("Missing entries in columns : ")
for col in cols:
    missing_entries = dataset[col].isnull().sum()
    if missing_entries > 0:
        print (col)
dataset['Age']=dataset['Age'].fillna(dataset['Age']).median()       
        
        


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