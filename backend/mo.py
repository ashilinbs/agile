import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('food_recommendation_dataset.csv')

# Check for missing values and handle them if necessary
df = df.dropna()  # Drop rows with missing values
print(f"Dataset shape after dropping missing values: {df.shape}")

# Splitting features and targets for calorie prediction
X = df[['Height (cm)', 'Weight (kg)', 'Age (years)', 'Gender_Male', 'Activity Level_Sedentary', 'Activity Level_Moderate', 'Activity Level_Active']]
y_calories = df['Calories (kcal)']

# Scaling the features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split for calorie prediction
X_train, X_test, y_train_calories, y_test_calories = train_test_split(X_scaled, y_calories, test_size=0.2, random_state=42)

# Train a Random Forest Regressor for predicting calories
calories_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
calories_model.fit(X_train, y_train_calories)

# Predict the calories and evaluate
y_pred_calories = calories_model.predict(X_test)
mae = mean_absolute_error(y_test_calories, y_pred_calories)
print(f'Mean Absolute Error for Calorie Prediction: {mae}')

# Identifying the correct food recommendation columns
food_columns = [col for col in df.columns if 'Recommended Food' in col]

# Check if the food recommendation columns exist
if not food_columns:
    print("No food recommendation columns found!")
else:
    print(f"Food columns: {food_columns}")
    y_food = df[food_columns].fillna(0)  # Handle missing values by filling with 0
    print(f"y_food sample after filling NaN values:\n{y_food.head()}")

    # Train-test split for food recommendation
    X_train_food, X_test_food, y_train_food, y_test_food = train_test_split(X_scaled, y_food, test_size=0.2, random_state=42)
    
    # Check if y_train_food is populated
    if not y_train_food.empty:
        food_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        food_model.fit(X_train_food, y_train_food)
        print("Food model trained successfully!")
        
        # Predict the food recommendation and evaluate
        y_pred_food = food_model.predict(X_test_food)
        food_accuracy = accuracy_score(y_test_food, y_pred_food)
        print(f'Accuracy for Food Recommendation: {food_accuracy * 100}%')
    else:
        print("y_train_food is still empty. Check the columns and data preprocessing.")
