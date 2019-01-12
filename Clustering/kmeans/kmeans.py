'''
    Clustering with scikit-learn
    
    Algorithm : K-means

'''
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# initialize logger
import logging
logging.basicConfig(level = logging.INFO,
                    format = '%(asctime)s - %(levelname)s - %(message)s')

# Read Dataset from file
dataset = pd.read_csv('Mall_Customers.csv')
logging.info('Dataset Shape : ' + str(dataset.shape) )

areMissingValuesPresent = dataset.isnull().values.any()
if areMissingValuesPresent:
    logging.warning('Missing Values Present in dataset ')

X = dataset.iloc[:, 1:].values
logging.info('X Shape : ' + str(X.shape) )



#  HANDLING CATEGORICAL DATA ######################################################################################
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()


X = pd.DataFrame(X)


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

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

print(X_train)
print(y_train)