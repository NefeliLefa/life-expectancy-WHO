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
df = pd.read_csv('LifeExpectancyData.csv') 

# Replacing the Null Values with mean values of the data
imputer=SimpleImputer(missing_values=np.nan,strategy='mean',fill_value=None)

df['Life Expectancy ']=imputer.fit_transform(df[['Life Expectancy ']])
df['Schooling']=imputer.fit_transform(df[['Schooling']])
df['Income composition of resources']=imputer.fit_transform(df[['Income composition of resources']])

#line plot with 95% confidence interval
plt.figure(figsize=(8,5))
plt.style.use('Solarize_Light2') 
sns.lineplot(x = df['Schooling'], y = df['Life Expectancy '], marker = 'o',errorbar=("ci", 95) ) #calculates mean for each year
plt.title("Average Life Expectancy ")
plt.show()

y=df['Life Expectancy ']
x=df[['Schooling']] #double [] to form dataframe

'''
scaler = StandardScaler()   # (x - mean(x)) / std(x) 
x = scaler.fit_transform(x) # scale input data x
'''

# split to training and testing
x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.2, random_state=42) 

Linear_model= LinearRegression()
Linear_model.fit(x_tr,y_tr)

#make predictions using testing data
predictions=Linear_model.predict(x_ts)

# calculate root mean squared error, R^2 score, theta0, theta1
print("RMSE linear with one parametre:", str(root_mean_squared_error(y_ts,predictions)))
print("R^2 score linear with one parametre:", str(r2_score(y_ts,predictions)))
print(f'Coefficients linear with one parametre: {Linear_model.coef_}')
print(f'Intercept linear with one parametre: {Linear_model.intercept_}')

# %% graphical representation
plt.figure(figsize=(8, 5))
plt.scatter(x_ts, y_ts, color="skyblue", label="Real/Testing data")  # Real data (Test Set)

plt.plot(x_ts, Linear_model.predict(x_ts), color="navy", label="Linear Regression")  # Linear Regression Model
plt.xlabel("Schooling")
plt.ylabel("Life Expectancy")
plt.title("Linear Regression Model for Life Expectancy ")
plt.legend()
plt.legend()
plt.grid()
plt.show()

# %% Multi-linear regression model

# select variables
features = ['Schooling', 'Income composition of resources']
target = 'Life Expectancy '

# delete NaN values
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# scale data because there are 2 variables (turn X into numpy array)
scaler = StandardScaler()  
X = scaler.fit_transform(X) 

# split to training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# transform numpy array to dataframe
X_test = pd.DataFrame(X_test, columns=features)
X_train = pd.DataFrame(X_train, columns=features)

# %% create model
model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

y_pred = model_multi.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'RMSE linear regression with two parameters: {rmse}')
print(f'R^2 linear regression with two parameters: {r2}')
print(f'Coefficients linear regression with two parameters: {model_multi.coef_}') #theta_1, theta_2
print(f'Intercept linear regression with two parameters: {model_multi.intercept_}') #theta_0

# %% 2D graphical representation

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# scatter plots 
sns.scatterplot(x=X_test['Schooling'], y=y_test, color='green', label='Actual', ax=ax[0]) # real/test data
sns.scatterplot(x=X_test['Schooling'], y=y_pred, color='darkviolet', label='Predicted', ax=ax[0]) # predictions
ax[0].set_title("Life Expectancy and Schooling")

sns.scatterplot(x=X_test['Income composition of resources'], y=y_test, color='green', label='Actual', ax=ax[1]) # real/test data
sns.scatterplot(x=X_test['Income composition of resources'], y=y_pred, color='darkviolet', label='Predicted', ax=ax[1]) # predictions
ax[1].set_title("Life Expectancy and Income Composition")

plt.show()

# %% 3D graphical representation
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test['Schooling'], X_test['Income composition of resources'], y_test, color='green', label='Actual')
ax.scatter(X_test['Schooling'], X_test['Income composition of resources'], y_pred, color='darkviolet', label='Predicted')

# regression surface
xx, yy = np.meshgrid(np.linspace(X_test['Schooling'].min(), X_test['Schooling'].max(), 10),
                     np.linspace(X_test['Income composition of resources'].min(), X_test['Income composition of resources'].max(), 10))
zz = model_multi.intercept_ + model_multi.coef_[0] * xx + model_multi.coef_[1] * yy
ax.plot_surface(xx, yy, zz, alpha=0.5, color='aqua')

ax.set_xlabel('Schooling')
ax.set_ylabel('Income composition of resources')
ax.set_zlabel('Life Expectancy')
ax.set_title("Linear Regression Model with two Variables")
plt.legend()
plt.show()

#%% Correlation matrix
'''
num_cols = ['Life Expectancy ', 'Adult Mortality', 'infant deaths', 'under-five deaths ', 'Alcohol',
 'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ', 'Polio', 'Total expenditure', 'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years', ' thinness 5-9 years', 'Income composition of resources', 'Schooling']
corr_matrix = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Heatmap of Correlation Matrix for Numerical Columns')
plt.show()
'''












