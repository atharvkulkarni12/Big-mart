import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


X = pd.read_csv("Train.csv")
X_test = pd.read_csv("Test.csv")

X_train = X.iloc[:, 0:11]
y_train = X.iloc[:, 11].values

X_train = X_train[['Item_Weight','Item_Fat_Content','Item_Visibility','Item_Type','Item_MRP','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type','Outlet_Type']]

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train[["Item_Weight"]])
X_train[["Item_Weight"]] = imputer.transform(X_train[["Item_Weight"]])

s = X_train["Outlet_Size"]

for i in range(0,8523):
    if pd.isnull(s[i]) == True  :
        s[i] = 'Medium'
    else :
        pass
    
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_1 = LabelEncoder()
X_train['Item_Fat_Content'] = labelencoder_1.fit_transform(X_train['Item_Fat_Content'])
labelencoder_2 = LabelEncoder()
X_train["Item_Type"] = labelencoder_2.fit_transform(X_train['Item_Type'])
labelencoder_3 = LabelEncoder()
X_train["Outlet_Location_Type"] = labelencoder_3.fit_transform(X_train['Outlet_Location_Type'])
labelencoder_4 = LabelEncoder()
X_train["Outlet_Type"] = labelencoder_4.fit_transform(X_train['Outlet_Type'])
labelencoder_5 = LabelEncoder()
X_train["Outlet_Size"] = labelencoder_5.fit_transform(X_train['Outlet_Size'])
'''
onehotencoder = OneHotEncoder(categorical_features=[1,3,6,7,8], sparse=False)
X_train_1 = onehotencoder.fit_transform(X_train_1)                             
p = X_train_1.toarray()
'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

#preparing test data set
X_test = X_test[['Item_Weight','Item_Fat_Content','Item_Visibility','Item_Type','Item_MRP','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type','Outlet_Type']]
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_test[["Item_Weight"]])
X_test[["Item_Weight"]] = imputer.transform(X_test[["Item_Weight"]])

q = X_test["Outlet_Size"]

for i in range(0,5681):
    if pd.isnull(q[i]) == True  :
        q[i] = 'Medium'
    else :
        pass

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_1 = LabelEncoder()
X_test['Item_Fat_Content'] = labelencoder_1.fit_transform(X_test['Item_Fat_Content'])
labelencoder_2 = LabelEncoder()
X_test["Item_Type"] = labelencoder_2.fit_transform(X_test['Item_Type'])
labelencoder_3 = LabelEncoder()
X_test["Outlet_Location_Type"] = labelencoder_3.fit_transform(X_test['Outlet_Location_Type'])
labelencoder_4 = LabelEncoder()
X_test["Outlet_Type"] = labelencoder_4.fit_transform(X_test['Outlet_Type'])
labelencoder_5 = LabelEncoder()
X_test["Outlet_Size"] = labelencoder_5.fit_transform(X_test['Outlet_Size'])

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_test = sc_X.fit_transform(X_test)
'''
onehotencoder = OneHotEncoder(categorical_features=[1,3,6,7,8], sparse=False)
X_test_1 = onehotencoder.fit_transform(X_test_1)                             
r = X_test_1.toarray()
'''
y_pred = regressor.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)

SampleSubmission_data = pd.read_csv('SampleSubmission.csv')

SampleSubmission_data['Item_Outlet_Sales'] = y_pred

SampleSubmission_data.to_csv('submission_3.csv', index=False)