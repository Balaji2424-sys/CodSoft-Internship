#----------------------------------------------- Section one to include libraries ---------------------------------------------------#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# to split data into two groups one for training and one for testing
from sklearn.model_selection import train_test_split

# to import linear regression
from sklearn.linear_model import LinearRegression

# imporving tools for better predictions
from sklearn.metrics import mean_squared_error, r2_score


#------------------------------------------------------ Section two reading data -------------------------------------------------#

data = pd.read_csv("advertising.csv") # read_excel() also used

#to verify the data are loaded (data.tail() // data.sample()) also used
print(data.head())

#------------------------------------------------------ Section three data exploration --------------------------------------------#

print(data.shape) # to know dimensions (rows & cols)

print(data.info()) # to know data types and missing values

print(data.describe()) # to generate summaries mean, std, min, max and etc

#-------------------------------------------------- Section four data visualization -------------------------------------------------#

# to visulize in scatter plots
sns.pairplot(data)
plt.show()

# to know the correlation b/w each values in cols & rows
sns.heatmap(data.corr(), annot=True)
plt.show()

#------------------------------------------------ Section five preprocessing ---------------------------------------------------------#

x = data.drop("Sales", axis=1) # axis = 1 (col) // axis = 0 (row)

y = data["Sales"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# here 0.2 helps splitting 20% data for testing and 80% for training
# random state = 42 helps regeneration of splits

# ------------------------------------------------ Section Six Model Training ------------------------------------------------------------#

#creating an empty model to train
model = LinearRegression()

#fitting the model according to the dimensions 
model.fit(x_train, y_train)

#-------------------------------------------------- Sectio seven Prediction and Evaluation ------------------------------------------------#

y_pred = model.predict(x_test)

print("R - Squared Score:", r2_score(y_test,y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
