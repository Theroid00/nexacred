from flask import Flask, request, jsonify
from flask_cors import CORS
import bcrypt
import re
from bson import ObjectId
from datetime import datetime
import traceback
import sys
import os

# Add ML path to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml'))

# Import our configuration
from config import (
    get_users_collection, 
    get_credit_scores_collection,
    create_user_document,
    create_credit_score_document
)

# Import ML components
try:
    from granite_agents import IBMGraniteFinancialAI
    from financial_assistant import NexaCredFinancialAssistant
    ML_AVAILABLE = True
    print("✅ ML components loaded successfully")
except ImportError as e:
    ML_AVAILABLE = False
    print(f"⚠️ ML components not available: {e}")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config['SECRET_KEY'] = 'nexacred_secret_key_2024'

# Initialize ML systems
if ML_AVAILABLE:
    try:
        granite_ai = IBMGraniteFinancialAI()
        financial_assistant = NexaCredFinancialAssistant()
        print("🧠 AI systems initialized successfully")
    except Exception as e:
        print(f"⚠️ Failed to initialize AI systems: {e}")
        granite_ai = None
        financial_assistant = None
else:
    granite_ai = None
    financial_assistant = None

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

@app.route('/api/ml/status', methods=['GET'])
def ml_status():
    """Get ML system status"""
    if not ML_AVAILABLE:
        return jsonify({
            'ml_available': False,
            'message': 'ML components not loaded'
        })

    try:
        status = {
            'ml_available': True,
            'granite_status': granite_ai.get_model_status() if granite_ai else None,
            'financial_assistant_available': financial_assistant is not None,
            'timestamp': datetime.utcnow().isoformat()
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'error': True,
            'message': f'Failed to get ML status: {str(e)}'
        }), 500

@app.route('/api/credit-score', methods=['POST'])
def calculate_credit_score_ml():
    """Calculate credit score using AI system"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': True, 'message': 'No data provided'}), 400

        user_id = data.get('user_id')
        customer_data = data.get('customer_data', {})

        if not user_id:
            return jsonify({'error': True, 'message': 'User ID required'}), 400

        if not ML_AVAILABLE or not granite_ai:
            # Fallback scoring
            score = 650  # Default score
            return jsonify({
                'success': True,
                'credit_score': score,
                'category': 'Good',
                'probability_of_default': 0.05,
                'risk_level': 'Medium',
                'explanation': 'Fallback scoring - ML system not available',
                'timestamp': datetime.utcnow().isoformat()
            })

        # Use Granite AI for credit analysis
        credit_result = granite_ai.analyze_credit_risk(customer_data)

        # Store result in database
        credit_scores_collection = get_credit_scores_collection()
        if credit_scores_collection:
            score_document = create_credit_score_document(
                user_id=user_id,
                credit_score=credit_result.credit_score,
                factors=credit_result.key_factors,
                recommendations=credit_result.recommendations
            )
            credit_scores_collection.insert_one(score_document)

        return jsonify({
            'success': True,
            'credit_score': credit_result.credit_score,
            'category': credit_result.risk_level,
            'probability_of_default': credit_result.probability_of_default,
            'risk_level': credit_result.risk_level,
            'key_factors': credit_result.key_factors,
            'recommendations': credit_result.recommendations,
            'explanation': credit_result.explanation,
            'confidence': credit_result.confidence,
            'timestamp': datetime.utcnow().isoformat()
        })

    except Exception as e:
        print(f"Credit score calculation error: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': True,
            'message': 'Failed to calculate credit score'
        }), 500

@app.route('/api/loan-recommendation', methods=['POST'])
def get_loan_recommendation():
    """Get personalized loan recommendation"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': True, 'message': 'No data provided'}), 400

        customer_data = data.get('customer_data', {})
        loan_type = data.get('loan_type', 'personal')

        if not ML_AVAILABLE or not granite_ai:
            # Fallback recommendation
            return jsonify({
                'success': True,
                'eligible': True,
                'loan_type': loan_type,
                'recommended_amount': 500000,
                'interest_rate': 12.0,
                'max_tenure_months': 60,
                'estimated_emi': 11122.22,
                'approval_probability': 0.75,
                'explanation': 'Fallback recommendation - ML system not available',
                'timestamp': datetime.utcnow().isoformat()
            })

        # Use Granite AI for loan recommendation
        loan_result = granite_ai.generate_loan_recommendation(customer_data, loan_type)
        loan_result['success'] = True
        loan_result['timestamp'] = datetime.utcnow().isoformat()

        return jsonify(loan_result)

    except Exception as e:
        print(f"Loan recommendation error: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': True,
            'message': 'Failed to generate loan recommendation'
        }), 500

@app.route('/api/fraud-check', methods=['POST'])
def check_fraud():
    """Check transaction for fraud indicators"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': True, 'message': 'No data provided'}), 400

        transaction_data = data.get('transaction_data', {})

        if not ML_AVAILABLE or not granite_ai:
            # Fallback fraud check
            return jsonify({
                'success': True,
                'fraud_probability': 0.05,
                'risk_indicators': [],
                'recommended_action': 'Allow transaction',
                'confidence': 0.70,
                'analysis': 'Fallback analysis - ML system not available',
                'timestamp': datetime.utcnow().isoformat()
            })

        # Use Granite AI for fraud detection
        fraud_result = granite_ai.detect_fraud(transaction_data)
        fraud_result['success'] = True
        fraud_result['timestamp'] = datetime.utcnow().isoformat()

        return jsonify(fraud_result)

    except Exception as e:
        print(f"Fraud check error: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': True,
            'message': 'Failed to check for fraud'
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
