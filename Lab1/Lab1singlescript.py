import numpy as np
import matplotlib as plt
import pandas as pd

data = pd.read_csv("L1Data.csv")

# just checking if I can print the data out using panda
# print(data)


#without the .values at the end, data is imported with pandas, creating a "pretty" data table.  
#using .values saves the information in a raw format
x = data[["Class", "Age", "Funds"]].values
y = data[["Sale"]].values

from sklearn.preprocessing import Imputer
#imputing the missing values...axis = 0 means by column
imputing_configuration = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputed_values = imputing_configuration.fit(x[:,[1,2]])

#this says to look at x, look at the column1 and column2 of x (age and funds), then replace
#the missing values with these imputed values
x[:,[1,2]] = imputed_values.transform(x[:,[1,2]])

print("the answer to one of the lab questions",x[7][2])




from sklearn.preprocessing import LabelEncoder
discreteCoder_x = LabelEncoder()

#replace the first column with the output of discreteCoder
#by using .fit_transform, we will encode the discrete column AND return the encoded labels
x[:,0] = discreteCoder_x.fit_transform(x[:,0])




from sklearn.preprocessing import OneHotEncoder
#the default for OneHotEncoder is to try and catergorize and pivot wider ALL the columns, 
#by specifiying column zero, we only want to apply it to the first column.

#we are creating a function using OneHotEncoder as a base to alter the first column in the entry
discreteCoder_x_dummies = OneHotEncoder(categorical_features = [0])
x = discreteCoder_x_dummies.fit_transform(x).toarray()

#the effect of this is to take a single column with categorical information, and create multiple columns
#that are all binary representations of the total number of categories in the categorical column data.
#in this case, there are four classes.  Now we have four columns, all binary.

#this is going to change the categorical data for our outcome variable
#we have create a new encoder each time
#since the outcome variable has only two categories, we dont have to worry about the weights of
#the categories, so we can just leave them as 1s and 0s
discreteCoder_y = LabelEncoder()
y = discreteCoder_y.fit_transform(y)



from sklearn.cross_validation import train_test_split
#we are creating a split between the training and testing data

#train_test_split will set aside 20% of the data for testing, and we have set a seed at 1693 for replication 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=1693)

#if we want to view any of the sets we just created
#print(y_train)


#feature scaling
from sklearn.preprocessing import StandardScaler

scale_x = StandardScaler()
x_train = scale_x.fit_transform(x_train)
x_test = scale_x.transform(x_test)