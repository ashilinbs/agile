import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import randint
import joblib

# Load the dataset
df = pd.read_csv("epi_r.csv")

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Check for NaN values in the target variable 'calories' and drop them
print(f"\nNaN values in target variable 'calories': {df['calories'].isna().sum()}")
df = df.dropna(subset=['calories'])

# Check the columns in the dataset to identify which features are available
print("\nColumns in the dataset:")
print(df.columns)

# Select relevant features for prediction
# We'll choose 'protein', 'fat', 'sodium', and 'rating' as our numerical features
# We'll include some categorical features like '#cakeweek', 'yuca', etc. to see if they influence calories

X = df[['protein', 'fat', 'sodium', 'rating', '#cakeweek', 'yuca', 'zucchini', 'cookbooks', 'leftovers', 'snack', 'snack week', 'turkey']]  # Modify based on available columns
y = df['calories']

# Check for NaN values in the feature set
print(f"\nNaN values in feature set: {X.isna().sum().sum()}")

# Define numeric and categorical features
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Define the preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing for numeric and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Sample a subset for faster training and testing
df_small = df.sample(n=1000, random_state=42)
X_small = df_small[['protein', 'fat', 'sodium', 'rating', '#cakeweek', 'yuca', 'zucchini', 'cookbooks', 'leftovers', 'snack', 'snack week', 'turkey']]  # Adjust based on available features
y_small = df_small['calories']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_small, y_small, test_size=0.2, random_state=42)

# Check dataset sizes
print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Build the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Parameter distribution for RandomizedSearchCV
param_dist = {
    'regressor__n_estimators': [100, 150],
    'regressor__max_depth': [10, 20],
    'regressor__min_samples_split': [2, 3],
    'regressor__min_samples_leaf': [1, 2]
}

# Run RandomizedSearchCV
random_search = RandomizedSearchCV(
    model, param_dist, n_iter=2, cv=3, n_jobs=-1, scoring='neg_mean_squared_error', random_state=42, verbose=2
)

print("\nPerforming randomized search...")
try:
    random_search.fit(X_train, y_train)
    print("\nRandomized search completed!")
except Exception as e:
    print(f"\nError during training: {e}")

# Best parameters from random search
print(f"\nBest parameters from random search: {random_search.best_params_}")

# Save the best model pipeline (preprocessor + regressor)
joblib.dump(random_search.best_estimator_, 'calorie_predictor_model.pkl')
print("\nModel saved as 'calorie_predictor_model.pkl'.")

# Predict on the test set
y_pred = random_search.best_estimator_.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Print evaluation metrics
print(f"\nModel Evaluation:")
print(f"R-squared (accuracy): {r2:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")

# Print some sample predictions vs actual values
print("\nSample predictions vs actual values:")
print(pd.DataFrame({'Actual': y_test.head().values, 'Predicted': y_pred[:5]}))
