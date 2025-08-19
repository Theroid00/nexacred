from pymongo import MongoClient
import os
from datetime import datetime

class Config:
    # MongoDB configuration
    MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/creditdb')
    DB_NAME = 'creditdb'
    USERS_COLLECTION = 'users'
    CREDIT_SCORES_COLLECTION = 'credit_scores'

class DatabaseConnection:
    _instance = None
    _client = None
    _db = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            try:
                self._client = MongoClient(Config.MONGO_URI)
                self._db = self._client[Config.DB_NAME]
                print(f"Connected to MongoDB: {Config.DB_NAME}")
                
                # Test the connection
                self._client.admin.command('ping')
                print("MongoDB connection successful!")
                
                # Create indexes for better performance
                self.create_indexes()
                
            except Exception as e:
                print(f"Failed to connect to MongoDB: {e}")
                self._client = None
                self._db = None

    def create_indexes(self):
        """Create database indexes for better performance"""
        if self._db is not None:
            try:
                # Create unique index on email for users collection
                self._db[Config.USERS_COLLECTION].create_index("email", unique=True)
                self._db[Config.USERS_COLLECTION].create_index("username", unique=True)
                
                # Create index on user_id for credit scores
                self._db[Config.CREDIT_SCORES_COLLECTION].create_index("user_id")
                
                print("Database indexes created successfully!")
            except Exception as e:
                print(f"Error creating indexes: {e}")

    def get_database(self):
        return self._db

    def get_collection(self, collection_name):
        if self._db is not None:
            return self._db[collection_name]
        return None

    def close_connection(self):
        if self._client:
            self._client.close()
            print("MongoDB connection closed")

# Utility functions
def get_db():
    """Get database instance"""
    db_connection = DatabaseConnection()
    return db_connection.get_database()

def get_users_collection():
    """Get users collection"""
    db_connection = DatabaseConnection()
    return db_connection.get_collection(Config.USERS_COLLECTION)

def get_credit_scores_collection():
    """Get credit scores collection"""
    db_connection = DatabaseConnection()
    return db_connection.get_collection(Config.CREDIT_SCORES_COLLECTION)

def create_user_document(username, email, password_hash):
    """Create a user document structure"""
    return {
        'username': username,
        'email': email,
        'password_hash': password_hash,
        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow(),
        'is_active': True,
        'credit_score': None,
        'profile': {
            'first_name': '',
            'last_name': '',
            'phone': '',
            'address': '',
            'date_of_birth': None
        }
    }

def create_credit_score_document(user_id, score, model_version='v1.0'):
    """Create a credit score document structure"""
    return {
        'user_id': user_id,
        'credit_score': score,
        'model_version': model_version,
        'calculated_at': datetime.utcnow(),
        'factors': {
            'payment_history': None,
            'credit_utilization': None,
            'length_of_credit_history': None,
            'types_of_credit': None,
            'new_credit_inquiries': None
        },
        'risk_assessment': 'pending'
    }
