from flask import Flask, request, jsonify
from flask_cors import CORS
import bcrypt
import re
from bson import ObjectId
from datetime import datetime
import traceback

# Import our configuration
from config import (
    get_users_collection, 
    get_credit_scores_collection,
    create_user_document,
    create_credit_score_document
)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config['SECRET_KEY'] = 'nexacred_secret_key_2024'

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """Validate password strength"""
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    return True, "Password is valid"

def hash_password(password):
    """Hash password using bcrypt"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed

def verify_password(password, hashed):
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'Backend running',
        'service': 'Nexacred Credit Scoring System',
        'version': '1.0.0',
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': True, 'message': 'No data provided'}), 400
        
        username = data.get('username', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        # Validation
        if not username or not email or not password:
            return jsonify({
                'error': True, 
                'message': 'Username, email, and password are required'
            }), 400
        
        if len(username) < 3:
            return jsonify({
                'error': True, 
                'message': 'Username must be at least 3 characters long'
            }), 400
        
        if not validate_email(email):
            return jsonify({
                'error': True, 
                'message': 'Invalid email format'
            }), 400
        
        is_valid, message = validate_password(password)
        if not is_valid:
            return jsonify({'error': True, 'message': message}), 400
        
        # Get users collection
        users_collection = get_users_collection()
        if users_collection is None:
            return jsonify({
                'error': True, 
                'message': 'Database connection failed'
            }), 500
        
        # Check if user already exists
        existing_user = users_collection.find_one({
            '$or': [
                {'email': email},
                {'username': username}
            ]
        })
        
        if existing_user:
            if existing_user['email'] == email:
                return jsonify({
                    'error': True, 
                    'message': 'Email already registered'
                }), 409
            else:
                return jsonify({
                    'error': True, 
                    'message': 'Username already taken'
                }), 409
        
        # Hash password
        password_hash = hash_password(password)
        
        # Create user document
        user_document = create_user_document(username, email, password_hash)
        
        # Insert user into database
        result = users_collection.insert_one(user_document)
        
        if result.inserted_id:
            return jsonify({
                'success': True,
                'message': 'User registered successfully',
                'user_id': str(result.inserted_id)
            }), 201
        else:
            return jsonify({
                'error': True,
                'message': 'Failed to register user'
            }), 500
            
    except Exception as e:
        print(f"Registration error: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': True,
            'message': 'Internal server error during registration'
        }), 500

@app.route('/login', methods=['POST'])
def login():
    """Login user"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': True, 'message': 'No data provided'}), 400
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        # Validation
        if not email or not password:
            return jsonify({
                'error': True, 
                'message': 'Email and password are required'
            }), 400
        
        if not validate_email(email):
            return jsonify({
                'error': True, 
                'message': 'Invalid email format'
            }), 400
        
        # Get users collection
        users_collection = get_users_collection()
        if users_collection is None:
            return jsonify({
                'error': True, 
                'message': 'Database connection failed'
            }), 500
        
        # Find user by email
        user = users_collection.find_one({'email': email})
        
        if not user:
            return jsonify({
                'error': True,
                'message': 'Invalid email or password'
            }), 401
        
        # Verify password
        if not verify_password(password, user['password_hash']):
            return jsonify({
                'error': True,
                'message': 'Invalid email or password'
            }), 401
        
        # Update last login
        users_collection.update_one(
            {'_id': user['_id']},
            {'$set': {'last_login': datetime.utcnow()}}
        )
        
        # Return success response
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': {
                'id': str(user['_id']),
                'username': user['username'],
                'email': user['email'],
                'credit_score': user.get('credit_score'),
                'created_at': user['created_at'].isoformat() if user.get('created_at') else None
            }
        }), 200
        
    except Exception as e:
        print(f"Login error: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': True,
            'message': 'Internal server error during login'
        }), 500

@app.route('/user/<user_id>', methods=['GET'])
def get_user(user_id):
    """Get user information"""
    try:
        # Validate user_id format
        if not ObjectId.is_valid(user_id):
            return jsonify({
                'error': True,
                'message': 'Invalid user ID format'
            }), 400
        
        users_collection = get_users_collection()
        if users_collection is None:
            return jsonify({
                'error': True, 
                'message': 'Database connection failed'
            }), 500
        
        # Find user
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        
        if not user:
            return jsonify({
                'error': True,
                'message': 'User not found'
            }), 404
        
        # Return user data (exclude password)
        return jsonify({
            'success': True,
            'user': {
                'id': str(user['_id']),
                'username': user['username'],
                'email': user['email'],
                'credit_score': user.get('credit_score'),
                'created_at': user['created_at'].isoformat() if user.get('created_at') else None,
                'is_active': user.get('is_active', True),
                'profile': user.get('profile', {})
            }
        }), 200
        
    except Exception as e:
        print(f"Get user error: {e}")
        return jsonify({
            'error': True,
            'message': 'Internal server error'
        }), 500

@app.route('/credit-score/<user_id>', methods=['GET'])
def get_credit_score(user_id):
    """Get user's credit score"""
    try:
        if not ObjectId.is_valid(user_id):
            return jsonify({
                'error': True,
                'message': 'Invalid user ID format'
            }), 400
        
        credit_scores_collection = get_credit_scores_collection()
        if credit_scores_collection is None:
            return jsonify({
                'error': True, 
                'message': 'Database connection failed'
            }), 500
        
        # Find latest credit score for user
        credit_score = credit_scores_collection.find_one(
            {'user_id': ObjectId(user_id)},
            sort=[('calculated_at', -1)]  # Get most recent
        )
        
        if not credit_score:
            return jsonify({
                'error': True,
                'message': 'No credit score found for this user'
            }), 404
        
        return jsonify({
            'success': True,
            'credit_score': {
                'score': credit_score['credit_score'],
                'model_version': credit_score.get('model_version'),
                'calculated_at': credit_score['calculated_at'].isoformat(),
                'risk_assessment': credit_score.get('risk_assessment'),
                'factors': credit_score.get('factors', {})
            }
        }), 200
        
    except Exception as e:
        print(f"Get credit score error: {e}")
        return jsonify({
            'error': True,
            'message': 'Internal server error'
        }), 500

@app.route('/calculate-credit-score/<user_id>', methods=['POST'])
def calculate_credit_score(user_id):
    """Calculate and store credit score for a user"""
    try:
        if not ObjectId.is_valid(user_id):
            return jsonify({
                'error': True,
                'message': 'Invalid user ID format'
            }), 400
        
        users_collection = get_users_collection()
        credit_scores_collection = get_credit_scores_collection()
        
        if users_collection is None or credit_scores_collection is None:
            return jsonify({
                'error': True, 
                'message': 'Database connection failed'
            }), 500
        
        # Check if user exists
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        if not user:
            return jsonify({
                'error': True,
                'message': 'User not found'
            }), 404
        
        # For demo purposes, generate a random credit score
        # In a real application, this would use the ML model
        import random
        calculated_score = random.randint(300, 850)
        
        # Create credit score document
        score_document = create_credit_score_document(
            ObjectId(user_id), 
            calculated_score
        )
        
        # Insert credit score
        result = credit_scores_collection.insert_one(score_document)
        
        # Update user's current credit score
        users_collection.update_one(
            {'_id': ObjectId(user_id)},
            {'$set': {'credit_score': calculated_score, 'updated_at': datetime.utcnow()}}
        )
        
        return jsonify({
            'success': True,
            'message': 'Credit score calculated successfully',
            'credit_score': calculated_score,
            'calculated_at': score_document['calculated_at'].isoformat()
        }), 200
        
    except Exception as e:
        print(f"Calculate credit score error: {e}")
        return jsonify({
            'error': True,
            'message': 'Internal server error'
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': True,
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': True,
        'message': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("Starting Nexacred Backend Server...")
    print("Make sure MongoDB is running on localhost:27017")
    print("Access the API at http://localhost:5000")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
