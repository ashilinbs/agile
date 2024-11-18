import pandas as pd
import numpy as np
import random

# Constants
n_samples = 500
foods = ['Chicken Salad', 'Grilled Chicken', 'Grilled Fish', 'Pasta', 'Pizza', 'Rice and Beans', 'Salad', 'Steak and Potatoes', 'Tuna Salad', 'Vegetable Stir Fry']
activity_levels = ['Sedentary', 'Moderate', 'Active']

# Function to generate a random activity level
def get_activity_level():
    return random.choice(activity_levels)

# Function to generate food recommendations (random selection from the food list)
def get_food_recommendations():
    food_dict = {food: 0 for food in foods}
    recommended_food = random.choice(foods)
    food_dict[recommended_food] = 1
    return food_dict

# Generating data
data = []
for _ in range(n_samples):
    # Random values for features
    height = random.randint(150, 190)  # cm
    weight = random.randint(50, 100)  # kg
    age = random.randint(18, 65)  # years
    calories = round(random.uniform(1500, 3500), 2)  # kcal
    gender = random.choice([0, 1])  # 0 for female, 1 for male
    activity_level = get_activity_level()

    # Get the food recommendations as one-hot encoded values
    food_recommendations = get_food_recommendations()

    # Creating a row of data
    row = {
        'Height (cm)': height,
        'Weight (kg)': weight,
        'Age (years)': age,
        'Calories (kcal)': calories,
        'Gender_Male': gender,
        'Activity Level_Sedentary': 1 if activity_level == 'Sedentary' else 0,
        'Activity Level_Moderate': 1 if activity_level == 'Moderate' else 0,
        'Activity Level_Active': 1 if activity_level == 'Active' else 0,
        **food_recommendations  # adding food recommendation columns
    }
    
    data.append(row)

# Creating DataFrame
df = pd.DataFrame(data)

# Save to CSV (optional)
df.to_csv('food_recommendation_dataset.csv', index=False)

# Display the first few rows of the dataset
print(df.head())
AIzaSyAgPyUBqQ4AWStR1MYlILvChVSilTG5Dqw
b7bef736a55940899eac9fdee2aa3818