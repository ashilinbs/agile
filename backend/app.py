from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_pymongo import PyMongo
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import bcrypt
from bson import ObjectId
from bson.errors import InvalidId
import logging
from werkzeug.utils import secure_filename
import requests

import datetime
import random

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)
CORS(app)


# Get today's date in UTC using timezone-aware datetime
today = datetime.datetime.now(datetime.timezone.utc).date()



# MongoDB configuration
app.config['MONGO_URI'] = 'mongodb+srv://ASHMI:Ashilin2010@cluster0.plfyt.mongodb.net/snackaroo?retryWrites=true&w=majority'
mongo = PyMongo(app)
# Check if the file has an allowed extension

# Test MongoDB connection
@app.route('/test_connection', methods=['GET'])
def test_connection():
    try:
        # Check if MongoDB is connected
        mongo.db.command('ping')
        return jsonify({'message': 'MongoDB connected successfully!'}), 200
    except Exception as e:
        return jsonify({'message': f'Error connecting to MongoDB: {str(e)}'}), 500

# Load the trained model
model = joblib.load('calorie_predictor_model.pkl')

# Function to recommend exercise based on predicted calories
def recommend_exercise(calories):
    if calories < 500:
        return "We recommend low-intensity exercises like walking or yoga."
    elif 500 <= calories < 1000:
        return "We recommend moderate exercises like jogging or cycling."
    else:
        return "We recommend high-intensity exercises like running or intense gym workouts."

# Helper function to convert ObjectId to string
def serialize_object_id(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

# User Registration Route
@app.route('/register', methods=['POST'])
def register_user():
    data = request.get_json()

    # Check if all required fields are present
    required_fields = ['name', 'email', 'password']
    for field in required_fields:
        if field not in data:
            return jsonify({'message': f'Missing required field: {field}'}), 400

    # Check if email already exists
    existing_user = mongo.db.users.find_one({'email': data['email']})
    if existing_user:
        return jsonify({'message': 'Email already exists'}), 400

    # Hash password
    hashed_password = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())

    # Create user document
    user = {
        'name': data['name'],
        'email': data['email'],
        'password': hashed_password,
        'predicted_history': [] ,
         'daily_missions': [] ,
          'meal_plan': [],  # Store weekly meal plans here
          'shopping_list': [] # Initialize predicted_history as an empty list
    }

    # Insert the new user into the database
    mongo.db.users.insert_one(user)

    return jsonify({'message': 'Registration successful'}), 200

# User Login Route
@app.route('/login', methods=['POST'])
def login_user():
    data = request.get_json()
    
    # Find the user by email
    user = mongo.db.users.find_one({'email': data['email']})
    
    if not user or not bcrypt.checkpw(data['password'].encode('utf-8'), user['password']):
        return jsonify({'message': 'Invalid credentials'}), 400
    
    # Convert ObjectId to string in user data
    user_data = {key: serialize_object_id(value) if isinstance(value, ObjectId) else value for key, value in user.items()}
    user_data.pop('password')  # Don't include password in the response
    
    return jsonify({'message': 'Login successful', 'user': user_data}), 200

# Fetch User Profile Route
def serialize_object_id(value):
    return str(value) if isinstance(value, ObjectId) else value

@app.route('/profile/<name>', methods=['GET'])
def get_user_profile(name):
    try:
        print(f"Fetching profile for: {name}")
        user = mongo.db.users.find_one({'name': name})
        
        if not user:
            print("User not found")
            return jsonify({'message': 'User not found'}), 404
        
        user_data = {key: str(value) if isinstance(value, ObjectId) else value for key, value in user.items()}
        user_data.pop('password', None)  # Exclude password for security
        
        return jsonify({'profile': user_data}), 200
    
    except Exception as e:
        print(f"Error fetching profile: {str(e)}")
        return jsonify({'message': f'Error fetching profile: {str(e)}'}), 500


@app.route('/predicted_history/<name>', methods=['GET'])
def get_user_predicted_history(name):
    try:
        user = mongo.db.users.find_one({'name': name})
        if not user:
            return jsonify({'message': 'User not found'}), 404
        
        predicted_history = user.get('predicted_history', [])
        return jsonify({'predicted_history': predicted_history}), 200
    except Exception as e:
        return jsonify({'message': f'Error fetching predicted history: {str(e)}'}), 500


# Endpoint to predict calories based on input features
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Prepare features from incoming data
    features = [
        data['protein'],
        data['fat'],
        data['sodium'],
        data['rating'],
        1 if data['#cakeweek'] == 'Yes' else 0,
        data['yuca'],
        data['zucchini'],
        data['cookbooks'],
        1 if data['leftovers'] == 'Yes' else 0,
        data['snack'],
        1 if data['snack week'] == 'Yes' else 0,
        data['turkey']
    ]
    
    # Create a DataFrame for prediction
    features_df = pd.DataFrame([features], columns=[ 
        'protein', 'fat', 'sodium', 'rating', '#cakeweek', 'yuca', 'zucchini',
        'cookbooks', 'leftovers', 'snack', 'snack week', 'turkey'
    ])
    
    # Make prediction
    prediction = model.predict(features_df)
    predicted_calories = prediction[0]
    
    # Get exercise recommendation
    exercise_recommendation = recommend_exercise(predicted_calories)

    user_id = data.get('name')
    if user_id:
        try:
            logging.info(f"Updating user with id: {user_id}")
            user = mongo.db.users.find_one({'name': user_id})
            if not user:
                logging.error(f"User with name {user_id} not found.")
                return jsonify({'error': 'User not found'}), 404

            logging.info(f"User found: {user}")
            # Ensure the user_id is valid ObjectId format
            
            result = mongo.db.users.update_one(
                {'name': user_id},
                {'$push': {'predicted_history': {'calories': predicted_calories, 'exercise': exercise_recommendation}}}
            )
                
            logging.info(f"Update result: {result.modified_count} document(s) modified.")
        except Exception as e:
            logging.error(f"Error updating prediction history: {str(e)}")

    # Return response with calories and exercise recommendation
    return jsonify({'calories': predicted_calories, 'exercise': exercise_recommendation})
@app.route('/upload-profile-picture', methods=['POST'])
def upload_profile_picture():
    data = request.get_json()
    image_url = data.get('imageUrl')
    name = data.get('name')

    # Check if the data is received correctly
    print("Received image URL:", image_url)  # Debug: print the received URL
    print("Received name:", name)  # Debug: print the name

    if not image_url or not name:
        return jsonify({'error': 'Missing image URL or name'}), 400

    # Assuming you're storing the user's data in a 'users' collection
    user = mongo.db.users.find_one({'name': name})
    
    if user:
        # Update the user's profile with the new image URL
        mongo.db.users.update_one(
            {'name': name},  # Find the user by name
            {'$set': {'image': image_url}}  # Set the new image URL
        )
        return jsonify({'imageUrl': image_url})  # Send the image URL back in the response
    else:
        return jsonify({'error': 'User not found'}), 404
@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback."""
    data = request.get_json()
    name = data.get('name')
    feedback_content = data.get('feedback')

    if not name or not feedback_content:
        return jsonify({'error': 'Missing name or feedback content'}), 400

    # Find the user by name
    user = mongo.db.users.find_one({'name': name})
    if not user:
        return jsonify({'error': 'User not found'}), 404

    feedback_entry = {
        '_id': str(ObjectId()),  # Unique ID for each feedback
        'feedback': feedback_content,
        'timestamp': datetime.datetime.now().isoformat()
    }

    # Add feedback to user's record
    mongo.db.users.update_one(
        {'name': name},
        {'$push': {'feedback': feedback_entry}}
    )

    return jsonify({'message': 'Feedback submitted successfully', 'feedback': feedback_entry}), 200


@app.route('/feedback/<name>', methods=['GET'])
def get_user_feedback(name):
    """Get feedback submitted by a specific user."""
    user = mongo.db.users.find_one({'name': name})
    if not user:
        return jsonify({'error': 'User not found'}), 404

    feedback = user.get('feedback', [])
    return jsonify({'feedback': feedback}), 200


@app.route('/feedback', methods=['GET'])
def get_all_feedback():
    """Get all feedback submitted by all users."""
    all_feedback = []
    users = mongo.db.users.find()

    for user in users:
        user_feedback = user.get('feedback', [])
        for entry in user_feedback:
            entry['user'] = user.get('name')  # Attach user name to feedback
            all_feedback.append(entry)

    return jsonify({'feedback': all_feedback}), 200


@app.route('/feedback/<user_name>/<feedback_id>', methods=['PUT'])
def edit_feedback(user_name, feedback_id):
    """Edit feedback submitted by the user."""
    data = request.get_json()
    new_feedback_content = data.get('feedback')

    if not new_feedback_content:
        return jsonify({'error': 'Missing new feedback content'}), 400

    user = mongo.db.users.find_one({'name': user_name})
    if not user:
        return jsonify({'error': 'User not found'}), 404

    feedback_list = user.get('feedback', [])
    updated = False

    for feedback in feedback_list:
        if feedback['_id'] == feedback_id:
            feedback['feedback'] = new_feedback_content
            feedback['timestamp'] = datetime.datetime.now().isoformat()  # Update timestamp
            updated = True
            break

    if not updated:
        return jsonify({'error': 'Feedback not found'}), 404

    mongo.db.users.update_one(
        {'name': user_name},
        {'$set': {'feedback': feedback_list}}
    )

    return jsonify({'message': 'Feedback updated successfully'}), 200


@app.route('/feedback/<user_name>/<feedback_id>', methods=['DELETE'])
def delete_feedback(user_name, feedback_id):
    """Delete feedback submitted by the user."""
    user = mongo.db.users.find_one({'name': user_name})
    if not user:
        return jsonify({'error': 'User not found'}), 404

    feedback_list = user.get('feedback', [])
    updated_feedback_list = [fb for fb in feedback_list if fb['_id'] != feedback_id]

    if len(feedback_list) == len(updated_feedback_list):
        return jsonify({'error': 'Feedback not found'}), 404

    mongo.db.users.update_one(
        {'name': user_name},
        {'$set': {'feedback': updated_feedback_list}}
    )

    return jsonify({'message': 'Feedback deleted successfully'}), 200

API_KEY = 'b7bef736a55940899eac9fdee2aa3818'
BASE_URL = "https://api.spoonacular.com/recipes/complexSearch"


def get_recipe_ingredients(meal_name):
    """Fetch recipe details and extract ingredients from Spoonacular API."""
    params = {
        'query': meal_name,
        'apiKey': API_KEY,
        'number': 1,  # Limit to one recipe per meal
    }
    
    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        return None
    
    data = response.json()
    if data.get("results"):
        recipe = data["results"][0]
        recipe_id = recipe["id"]
        ingredients_url = f"https://api.spoonacular.com/recipes/{recipe_id}/ingredientWidget.json"
        ingredients_params = {'apiKey': API_KEY}
        ingredients_response = requests.get(ingredients_url, params=ingredients_params)
        
        if ingredients_response.status_code == 200:
            ingredients_data = ingredients_response.json()
            ingredients = []
            for ingredient in ingredients_data["ingredients"]:
                ingredient_details = {
                    'name': ingredient["name"],
                    'category': ingredient.get("category", "Miscellaneous"),
                    'purchased': False
                }
                ingredients.append(ingredient_details)
            return ingredients
    return None


@app.route('/generate_shopping_list', methods=['POST'])
def generate_shopping_list():
    """Generate a shopping list based on selected meals for a specific user."""
    data = request.get_json()
    name = data.get('name')  # User's name
    meals = data.get('meals')  # List of meals selected by the user
    
    if not name or not meals:
        return jsonify({'error': 'Name and meals are required'}), 400
    
    shopping_list = []
    
    # Fetch ingredients for each meal and add to the shopping list
    for meal in meals:
        ingredients = get_recipe_ingredients(meal)
        if ingredients:
            shopping_list.extend(ingredients)
        else:
            return jsonify({'error': f'Could not fetch ingredients for {meal}'}), 404
    
    # Save the shopping list in the user's document in the database
    mongo.db.users.update_one(
        {'name': name},
        {'$set': {'shopping_list': shopping_list}}
    )
    
    return jsonify({'message': 'Shopping list generated successfully'}), 200


@app.route('/get_shopping_list', methods=['GET'])
def get_shopping_list():
    """Fetch the shopping list for a specific user."""
    name = request.args.get('name')
    
    if not name:
        return jsonify({'error': 'Name is required'}), 400
    
    user = mongo.db.users.find_one({'name': name})
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({'shopping_list': user.get('shopping_list', [])}), 200


@app.route('/update_shopping_item', methods=['POST'])
def update_shopping_item():
    """Update the purchased status of an item in the shopping list."""
    data = request.get_json()
    name = data.get('name')
    ingredient_name = data.get('ingredient_name')
    purchased = data.get('purchased', False)
    
    if not name or not ingredient_name:
        return jsonify({'error': 'Name and ingredient name are required'}), 400
    
    user = mongo.db.users.find_one({'name': name})
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Find and update the ingredient's purchased status
    mongo.db.users.update_one(
        {'name': name, 'shopping_list.name': ingredient_name},
        {'$set': {'shopping_list.$.purchased': purchased}}
    )
    
    return jsonify({'message': 'Shopping list updated successfully'}), 200




if __name__ == '__main__':
    app.run(debug=True)
