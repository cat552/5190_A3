import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Set up plot fonts to support Chinese if needed (optional, not used here)
# We skip this since we use English only

train_data = pd.DataFrame({
    'Height': [175, 182, 170, 178, 184, 173, 179, 176, 171, 180, 177, 181, 172, 183, 174],  # Height in cm
    'Weight': [57, 65, 55, 58, 75, 58, 63, 65, 53, 60, 62, 71, 57, 67, 60]              # Weight in kg
})

# Assign gender based on height
train_data['Gender'] = np.where(train_data['Height'] < 175, 'Female', 'Male')

female_train_data = train_data[train_data['Gender'] == 'Female']
male_train_data = train_data[train_data['Gender'] == 'Male']

# Training data
X_female_train = female_train_data[['Height']]  # Female features
y_female_train = female_train_data['Weight']    # Female target
X_male_train = male_train_data[['Height']]      # Male features
y_male_train = male_train_data['Weight']        # Male target

# Train models
female_model = LinearRegression()
female_model.fit(X_female_train, y_female_train)

male_model = LinearRegression()
male_model.fit(X_male_train, y_male_train)

# Print model parameters
print("Female Model - Regression Coefficient (Slope):", female_model.coef_[0])
print("Female Model - Intercept:", female_model.intercept_)
print("\nMale Model - Regression Coefficient (Slope):", male_model.coef_[0])
print("Male Model - Intercept:", male_model.intercept_)

# Predict on training set
predicted_weight_female_train = female_model.predict(X_female_train)
predicted_weight_male_train = male_model.predict(X_male_train)

# Calculate MAE for training set
mae_female_train = mean_absolute_error(y_female_train, predicted_weight_female_train)  # Female MAE
mae_male_train = mean_absolute_error(y_male_train, predicted_weight_male_train)        # Male MAE
print("\nFemale Training Set - Mean Absolute Error (MAE):", mae_female_train)
print("Male Training Set - Mean Absolute Error (MAE):", mae_male_train)

# Test data
test_heights = np.array([177, 174, 179, 172, 181, 175, 178, 173, 180, 176])  # Heights in cm
test_weights = np.array([60, 60, 64, 57, 66, 63, 66, 56, 62, 59])            # Weights in kg

# Create test DataFrame and assign gender
test_data = pd.DataFrame({
    'Height': test_heights,
    'Weight': test_weights
})
test_data['Gender'] = np.where(test_data['Height'] < 175, 'Female', 'Male')

# Split into female and male test sets
female_test_data = test_data[test_data['Gender'] == 'Female']
male_test_data = test_data[test_data['Gender'] == 'Male']

# Features and targets for test set
X_female_test = female_test_data[['Height']]  # Female features
y_female_test = female_test_data['Weight']    # Female target
X_male_test = male_test_data[['Height']]      # Male features
y_male_test = male_test_data['Weight']        # Male target

# Predict on test set
predicted_weight_female_test = female_model.predict(X_female_test)
predicted_weight_male_test = male_model.predict(X_male_test)

# Calculate MAE for test set
mae_female_test = mean_absolute_error(y_female_test, predicted_weight_female_test)  # Female MAE
mae_male_test = mean_absolute_error(y_male_test, predicted_weight_male_test)        # Male MAE
print("\nFemale Test Set - Mean Absolute Error (MAE):", mae_female_test)
print("Male Test Set - Mean Absolute Error (MAE):", mae_male_test)

# Output test data
print("\nTest Data:")
print(test_data)

# Plot results
plt.scatter(female_train_data['Height'], female_train_data['Weight'], color='pink', label='Female Training Data')
plt.scatter(male_train_data['Height'], male_train_data['Weight'], color='blue', label='Male Training Data')
plt.scatter(female_test_data['Height'], female_test_data['Weight'], color='red', label='Female Test Data')
plt.scatter(male_test_data['Height'], male_test_data['Weight'], color='green', label='Male Test Data')

# Female regression line
all_female_heights = np.linspace(min(female_train_data['Height']), max(female_train_data['Height']), 100)
all_female_weights = female_model.predict(all_female_heights.reshape(-1, 1))
plt.plot(all_female_heights, all_female_weights, color='pink', linestyle='--', label='Female Regression Line')

# Male regression line
all_male_heights = np.linspace(min(male_train_data['Height']), max(male_train_data['Height']), 100)
all_male_weights = male_model.predict(all_male_heights.reshape(-1, 1))
plt.plot(all_male_heights, all_male_weights, color='blue', linestyle='--', label='Male Regression Line')

plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Height vs Weight with Regression Lines')
plt.legend()
plt.grid(True)
plt.show()