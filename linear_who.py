# Import libraries
import pandas as pd # basic functions for dataset import, preprocessing etc.
import numpy as np # basic mathematical functions
import os # handle path 
import matplotlib.pyplot as plt # plots
import seaborn as sns # statistical data visualization
from sklearn.preprocessing import StandardScaler # scale input 
from sklearn.model_selection import train_test_split # for splitting data to train and test
from sklearn.linear_model import LinearRegression #train Linear Regression model
from sklearn.impute import SimpleImputer #replace null values with mean
from sklearn.metrics import root_mean_squared_error, r2_score #evaluate model

# Set the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# %% Import data and processing
df = pd.read_csv('LifeExpectancyData.csv') #import data
#selected_countries = ["Greece"]
#selected_countries = ["Greece" ,"Italy", "Spain", "Portugal", "Cyprus","Malta","France"]
#df = df[df["Country"].isin(selected_countries)]

# Replacing the Null Values with mean values of the data
imputer=SimpleImputer(missing_values=np.nan,strategy='mean',fill_value=None)
df['Life Expectancy ']=imputer.fit_transform(df[['Life Expectancy ']])

#line plot with 95% confidence interval
plt.figure(figsize=(8,5))
plt.style.use('Solarize_Light2') 
sns.lineplot(x = df['Year'], y = df['Life Expectancy '], marker = 'o',errorbar=("ci", 95) ) #calculates mean for each year
plt.title("Average Life Expectancy ")
plt.show()
'''
#line plot without seaborn library
grouped_df = df.groupby('Year')['Life Expectancy '].mean()
plt.plot(grouped_df.index, grouped_df.values)
plt.title('Average Life Expectancy by Year')
plt.xlabel("Year")
plt.ylabel("Life Expectancy")
plt.show()
'''
# %% pre-Processing
y=df['Life Expectancy ']
x=df[['Year']] #double [] to form dataframe

'''
scaler = StandardScaler()   # (x - mean(x)) / std(x) 
x = scaler.fit_transform(x) # scale input data x
'''

# split to training and testing
x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.2, random_state=42) 

# %% Parametric Model, linear regression
Linear_model= LinearRegression()
Linear_model.fit(x_tr,y_tr)

#make predictions using testing data
predictions=Linear_model.predict(x_ts)

# calculate root mean squared error,  R^2 score, theta0, theta1
print("RMSE:", str(root_mean_squared_error(y_ts,predictions)))
print("R^2 score:", str(r2_score(y_ts,predictions)))
print(Linear_model.coef_)
print(Linear_model.intercept_)

# %% graphical representation
plt.figure(figsize=(8, 5))
plt.scatter(x_ts, y_ts, color="skyblue", label="Real/Testing data")  # Real/ Test Data
plt.plot(x, Linear_model.predict(x), color="navy", label="Linear Regression")  # Linear Regression Model
plt.xlabel("Year")
plt.ylabel("Life Expectancy")
plt.title("Linear Regression Model for Life Expectancy ")
plt.legend()
plt.grid()
plt.show()
