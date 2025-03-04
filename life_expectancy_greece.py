import os                         # handle path 
import pandas as pd               # basic functions for dataset import, preprocessing etc.
import matplotlib.pyplot as plt   # plots
from matplotlib import style 
import numpy as np                # basic mathematical functions
from sklearn.impute import SimpleImputer #replace null values with mean

# Set working directory 
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# %% Import data

df = pd.read_csv('LifeExpectancyData.csv') # import data 
#df.isnull().sum() #there are 10 na in Life Expectency
imputer=SimpleImputer(missing_values=np.nan,strategy='mean',fill_value=None) #create imputer
df['Life Expectancy ']=imputer.fit_transform(df[['Life Expectancy ']]) #fill na values with mean

# %% Get info for Greece
selected_countries = ["Greece"]
small_df = df[df["Country"].isin(selected_countries)]

x = small_df['Year'].values #set x
y = small_df['Life Expectancy '].values #set y

# %% Creat plot for Life Expectancy in Greece over years
plt.style.use('Solarize_Light2') 
plt.figure().canvas.manager.set_window_title("Life Expectancy in Greece")
plt.plot(x,y, 'o')
plt.ylabel('Life Expectancy in Greece')
plt.xlabel('Year')
plt.tight_layout()
plt.show()

# %% Polyonomial Regression

x_pol = small_df["Year"].values  
y_pol = small_df['Life Expectancy '].values  

# Degree of Polyonomial Regression
degree = 7
coefficients = np.polyfit(x_pol, y_pol, degree)  
polynomial = np.poly1d(coefficients)  # Polyonomial Function

# %% Creat plot  for Life Expectancy in Greece over years 
x_curve = np.linspace(x_pol.min(), x_pol.max(), 100)
y_curve = polynomial(x_curve)

# creat graph
plt.figure(figsize=(8, 5))
plt.scatter(x_pol, y_pol, color="blue", label="Data")
plt.plot(x_curve, y_curve, color="blue")
plt.xlabel("Year")
plt.ylabel("Life Expectancy")
plt.title("Polyonomial Regression for Life Expectancy in Greece")
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,5))

