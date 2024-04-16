#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 
import seaborn as sns

mpg = pd.read_csv('auto-mpg.csv')

mpg.head()


# In[4]:


mpg.info()


# In[29]:


mpg [ "origin"] = mpg ["origin"].astype("object")
mpg.info()


# In[30]:


# changing data types of variable
mpg['horsepower'].unique()


# In[32]:


mpg.query('horsepower == "?"')


# In[33]:


mpg [ "horsepower"] = pd.to_numeric (mpg["horsepower"], errors="coerce")
mpg [ "horsepower"] = mpg["horsepower"].fillna (mpg [ "horsepower"].mean())
mpg.info()


# In[34]:


mpg.describe()


# In[10]:


sns.histplot(mpg['mpg'])


# In[11]:


sns.pairplot(mpg, corner =True)


# In[12]:


sns.barplot(data=mpg, x="origin" , y="mpg")


# In[13]:


sns.heatmap(
            mpg.corr(numeric_only =True),
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
            annot=True
           )


# In[37]:


mpg.head()


# In[51]:


from sklearn.model_selection import train_test_split

# Assuming 'mpg' and 'weight' are in the DataFrame and 'weight' will be used first
X = mpg[['weight']]  # DataFrame with the predictor
y = mpg['mpg']       # Series with the target

# Split the data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# In[52]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create a linear regression model object
model_weight = LinearRegression()

# Fit the model to the training data
model_weight.fit(X_train, y_train)

# Predict on the test set
y_pred = model_weight.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")




# In[53]:


# Adding more features to the model
features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year']
X = mpg[features]  # Update X with more features

# Re-split the dataset to include these new features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the multiple regression model
model_full = LinearRegression()
model_full.fit(X_train, y_train)
y_pred_full = model_full.predict(X_test)

# Evaluate the model
mse_full = mean_squared_error(y_test, y_pred_full)
r2_full = r2_score(y_test, y_pred_full)

print(f"Full Model Mean Squared Error: {mse_full:.2f}")
print(f"Full Model R^2 Score: {r2_full:.2f}")


# In[54]:


import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Calculate residuals
residuals = y_test - y_pred_full

# Plotting residuals
plt.scatter(y_pred_full, residuals)
plt.title('Residuals vs Fitted Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='red', linestyle='--')
plt.show()

# Q-Q plot for normality
sm.qqplot(residuals, line='s')
plt.title('Q-Q Plot of Residuals')
plt.show()

# Check the distribution of residuals
sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals')
plt.show()


# In[55]:


from sklearn.metrics import mean_absolute_error

# Compute predictions on the test set
y_pred_final = model_full.predict(X_test)

# Calculate R^2 and MAE for the final model
r2_final = r2_score(y_test, y_pred_final)
mae_final = mean_absolute_error(y_test, y_pred_final)

print(f"Final Model R^2 Score: {r2_final:.2f}")
print(f"Final Model Mean Absolute Error: {mae_final:.2f}")



# In[56]:


# Get the coefficient for 'model year' from the model
model_year_coefficient = model_full.coef_[features.index('model year')]

print(f"Coefficient for one-year increase in model year: {model_year_coefficient:.3f} mpg")


# In[57]:


from sklearn.linear_model import Ridge

# Define and fit the ridge regression model
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Score the ridge model on the test set
y_pred_ridge = ridge.predict(X_test)
r2_ridge = r2_score(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)

print(f"Ridge Regression R^2 Score: {r2_ridge:.2f}")
print(f"Ridge Regression Mean Absolute Error: {mae_ridge:.2f}")

# Compare the performance improvement if any
improvement_r2 = r2_ridge - r2_final
print(f"Improvement in R^2 with Ridge over regular regression: {improvement_r2:.4f}")


# In[58]:


# Calculate the correlation between 'weight' and 'mpg'
correlation = mpg['weight'].corr(mpg['mpg'])

print(f"The correlation between a car's 'weight' and 'mpg' is: {correlation:.3f}")


# In[ ]:




