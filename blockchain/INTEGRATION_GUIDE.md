# Blockchain Integration Guide

This guide explains how the NexaCred blockchain components integrate with the backend and frontend.

## Overview

The NexaCred platform uses two main smart contracts:

1. **CreditScore.sol** - Manages credit scores with audit trail and fraud protection
2. **NexaCred.sol** - Handles peer-to-peer lending operations

## Smart Contract Capabilities

### CreditScore Contract

**What it does:**
- Stores credit scores for users with complete history
- Tracks who updated scores and when
- Provides fraud flagging system
- Maintains immutable audit trail

**Key Functions for Backend:**
- `getCreditScore(address user)` - Get current credit score
- `updateScore(address user, uint256 score, string reason)` - Update score from ML service
- `getScoreChangeCount(address user)` - Get number of score updates
- `isFraud(address user)` - Check if user is flagged for fraud

**Integration Points:**
```python
# In Flask backend
blockchain.update_credit_score(user_wallet_address, ml_calculated_score)
current_score = blockchain.get_user_credit_score(user_wallet_address)
```

### NexaCred Lending Contract

**What it does:**
- Manages loan requests from borrowers
- Handles loan funding from lenders  
- Processes repayments and tracks defaults
- Maintains user reputation scores

**Key Functions for Backend:**
- `requestLoan(amount, rate, duration, purpose)` - Create new loan request
- `fundLoan(loanId)` - Fund an existing loan
- `repayLoan(loanId)` - Process loan repayment
- `getUserProfile(address)` - Get user lending statistics
- `getActiveLoans()` - List all available loans

**Integration Points:**
```python
# In Flask backend
loan_id = blockchain.create_loan_request(amount, rate, duration, purpose)
user_stats = blockchain.get_user_profile(user_wallet_address)
active_loans = blockchain.get_active_loans()
```

## Backend Integration

### 1. Installation

Add to `backend/requirements.txt`:
```
web3==6.0.0
python-dotenv==1.0.0
```

### 2. Configuration

In `backend/config.py`, add:
```python
import os
from blockchain.web3_integration import create_development_client, create_production_client

class Config:
    # Existing MongoDB config...
    
    # Blockchain configuration
    BLOCKCHAIN_ENABLED = os.getenv('BLOCKCHAIN_ENABLED', 'false').lower() == 'true'
    BLOCKCHAIN_NETWORK = os.getenv('BLOCKCHAIN_NETWORK', 'development')

def get_blockchain_client():
    if Config.BLOCKCHAIN_NETWORK == 'development':
        return create_development_client()
    else:
        return create_production_client()
```

### 3. Flask App Integration

In `backend/app.py`, add these new endpoints:

```python
from blockchain.web3_integration import create_development_client

# Initialize blockchain client
blockchain = create_development_client()

@app.route('/blockchain/loan-request', methods=['POST'])
def create_blockchain_loan():
    """Create loan request on blockchain"""
    data = request.get_json()
    
    result = blockchain.create_loan_request(
        borrower_address=data['user_wallet'],
        amount_eth=data['amount'],
        interest_rate=data['interest_rate'],
        duration_days=data['duration'],
        purpose=data['purpose']
    )
    
    return jsonify(result)

@app.route('/blockchain/credit-score/<user_id>', methods=['POST'])
def update_blockchain_credit_score(user_id):
    """Update credit score on blockchain after ML calculation"""
    # Get user's wallet address from database
    user = get_user_by_id(user_id)
    
    # Calculate credit score with ML model
    ml_score = calculate_credit_score_with_ml(user_id)
    
    # Update on blockchain
    result = blockchain.update_credit_score(
        user_address=user['wallet_address'],
        new_score=ml_score,
        reason=f"ML model calculation - {datetime.now()}"
    )
    
    return jsonify(result)

@app.route('/blockchain/active-loans', methods=['GET'])
def get_blockchain_loans():
    """Get active loans for marketplace"""
    result = blockchain.get_active_loans()
    return jsonify(result)
```

## Frontend Integration

### 1. Event Listening

Add to frontend JavaScript:
```javascript
// Listen for blockchain events
const API_BASE = 'http://localhost:5000';

async function loadActiveLoans() {
    try {
        const response = await fetch(`${API_BASE}/blockchain/active-loans`);
        const data = await response.json();
        
        if (data.success) {
            displayLoans(data.loans);
        }
    } catch (error) {
        console.error('Failed to load loans:', error);
    }
}

async function requestLoan(loanData) {
    try {
        const response = await fetch(`${API_BASE}/blockchain/loan-request`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(loanData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            showSuccessMessage(`Loan request created! Transaction: ${result.transaction_hash}`);
        } else {
            showErrorMessage(result.error);
        }
    } catch (error) {
        showErrorMessage('Failed to create loan request');
    }
}
```

### 2. User Wallet Integration

To fully integrate with blockchain, users need wallet addresses. Add to user registration:

```javascript
// Frontend wallet connection
async function connectWallet() {
    if (typeof window.ethereum !== 'undefined') {
        try {
            const accounts = await window.ethereum.request({
                method: 'eth_requestAccounts'
            });
            
            return accounts[0]; // User's wallet address
        } catch (error) {
            console.error('Failed to connect wallet:', error);
            return null;
        }
    } else {
        alert('Please install MetaMask to use blockchain features');
        return null;
    }
}
```

## Deployment Steps

### 1. Local Development

```bash
# Install dependencies
cd blockchain
npm install

# Start local blockchain (in separate terminal)
npx hardhat node

# Deploy contracts to local network
npm run deploy:local
```

### 2. Testnet Deployment

```bash
# Configure environment
cp .env.example .env
# Edit .env with your Infura key and private key

# Deploy to Sepolia testnet
npm run deploy:testnet

# Verify contracts on Etherscan
npm run verify:testnet
```

### 3. Production Deployment

```bash
# Configure mainnet environment variables
# Deploy to Ethereum mainnet
npm run deploy:mainnet

# Verify on Etherscan
npm run verify:mainnet
```

## Security Considerations

### 1. Private Key Management
- Never commit private keys to git
- Use environment variables or secure key management
- Use different keys for different environments

### 2. Access Control
- Only authorized addresses can update credit scores
- Platform admin controls fee rates and emergency pause
- KYC verification required for lending

### 3. Testing
- Test all functions on testnet before mainnet
- Monitor gas costs and optimize transactions
- Implement proper error handling for failed transactions

## Current Status & Missing Features

### ✅ Complete:
- Smart contract structure and core functionality
- Basic blockchain integration methods
- Development/testnet deployment setup
- Clear integration documentation

### ⚠️ Needs Implementation:
- Actual Web3 transaction execution (currently simulated)
- User wallet management in frontend
- Event monitoring for real-time updates
- Gas optimization and error handling
- Integration tests with actual contracts

### 🚀 Future Enhancements:
- Multi-signature wallet support
- Layer 2 integration (Polygon, Arbitrum)
- Cross-chain compatibility
- Advanced fraud detection algorithms
- Automated market making for loan rates

## Quick Test

Run the blockchain integration test:
```bash
cd blockchain
python web3_integration.py
```

This will test all integration methods and show you what works.
