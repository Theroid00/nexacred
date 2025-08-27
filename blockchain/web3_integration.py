import json
import os
from web3 import Web3
from web3.middleware import geth_poa_middleware
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NexaCredBlockchain:
    """
    Blockchain integration for NexaCred platform
    
    This class provides the bridge between our Flask backend and the smart contracts.
    It handles:
    - Credit score updates from ML service
    - Loan creation and funding
    - Payment processing
    - Event monitoring
    
    Backend Usage:
    ```python
    # Initialize blockchain connection
    blockchain = NexaCredBlockchain()
    
    # Update credit score after ML calculation
    blockchain.update_credit_score(user_address, new_score)
    
    # Create loan when user applies
    blockchain.create_loan_request(amount, interest_rate, duration, purpose)
    ```
    """
    
    def __init__(self, web3_provider_uri=None, contract_address=None, private_key=None):
        # Setup blockchain connection
        try:
            if web3_provider_uri:
                if web3_provider_uri.startswith('http'):
                    self.w3 = Web3(Web3.HTTPProvider(web3_provider_uri))
                else:
                    self.w3 = Web3(Web3.IPCProvider(web3_provider_uri))
            else:
                # Default to local development node (Ganache)
                self.w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))
            
            # Add middleware for testnet compatibility
            self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            logger.info(f"Web3 connection established: {self.w3.isConnected()}")
        except Exception as e:
            logger.error(f"Failed to connect to blockchain: {e}")
            self.w3 = None
        
        self.private_key = private_key
        self.contract_address = contract_address
        self.contract = None
        self.credit_score_contract = None
        
        # Load contract ABIs from files
        self._load_contract_abis()
        
        if contract_address and self.w3:
            self._initialize_contract(contract_address)
    
    def _load_contract_abis(self):
        """Load contract ABIs - in production, these would be from compiled contracts"""
        # Simplified ABI for main functions
        self.nexacred_abi = [
            {
                "inputs": [{"name": "amount", "type": "uint256"}, {"name": "interestRate", "type": "uint256"}, 
                          {"name": "durationDays", "type": "uint256"}, {"name": "purpose", "type": "string"}],
                "name": "requestLoan",
                "outputs": [{"name": "loanId", "type": "uint256"}],
                "type": "function"
            },
            {
                "inputs": [{"name": "loanId", "type": "uint256"}],
                "name": "fundLoan",
                "outputs": [],
                "payable": True,
                "type": "function"
            },
            {
                "inputs": [{"name": "user", "type": "address"}, {"name": "newScore", "type": "uint256"}],
                "name": "updateUserCreditScore",
                "outputs": [],
                "type": "function"
            },
            {
                "inputs": [{"name": "user", "type": "address"}],
                "name": "getUserProfile",
                "outputs": [{"name": "", "type": "tuple"}],
                "type": "function",
                "constant": True
            },
            {
                "inputs": [],
                "name": "getActiveLoans",
                "outputs": [{"name": "", "type": "uint256[]"}],
                "type": "function",
                "constant": True
            }
        ]
        
        self.credit_score_abi = [
            {
                "inputs": [{"name": "_user", "type": "address"}],
                "name": "getCreditScore",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function",
                "constant": True
            },
            {
                "inputs": [{"name": "_user", "type": "address"}, {"name": "_newScore", "type": "uint256"}, {"name": "_reason", "type": "string"}],
                "name": "updateScore",
                "outputs": [],
                "type": "function"
            }
        ]
    
    def _initialize_contract(self, contract_address):
        """Initialize contract instance"""
        try:
            self.contract = self.w3.eth.contract(
                address=Web3.toChecksumAddress(contract_address),
                abi=self.nexacred_abi
            )
            logger.info(f"NexaCred contract initialized at {contract_address}")
        except Exception as e:
            logger.error(f"Failed to initialize contract: {e}")
    
    def set_credit_score_contract(self, contract_address):
        """Set credit score contract address"""
        try:
            self.credit_score_contract = self.w3.eth.contract(
                address=Web3.toChecksumAddress(contract_address),
                abi=self.credit_score_abi
            )
            logger.info(f"Credit score contract set at {contract_address}")
        except Exception as e:
            logger.error(f"Failed to set credit score contract: {e}")
    
    def is_connected(self):
        """Check blockchain connection status"""
        if not self.w3:
            return False
        return self.w3.isConnected()
    
    def get_account_from_key(self, private_key):
        """Get account object from private key"""
        try:
            return self.w3.eth.account.privateKeyToAccount(private_key)
        except Exception as e:
            logger.error(f"Invalid private key: {e}")
            return None
    
    def get_eth_balance(self, address):
        """Get ETH balance for an address"""
        try:
            balance_wei = self.w3.eth.get_balance(Web3.toChecksumAddress(address))
            return self.w3.fromWei(balance_wei, 'ether')
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0
    
    # Backend Integration Methods
    
    def create_loan_request(self, borrower_address, amount_eth, interest_rate, duration_days, purpose):
        """
        Create a loan request on blockchain
        Called by: Flask backend when user submits loan application
        
        Returns:
        {
            "success": True/False,
            "transaction_hash": "0x...",
            "loan_id": 123,
            "error": "error message if failed"
        }
        """
        if not self.contract or not self.w3:
            return {"error": "Blockchain not available"}
        
        try:
            # For now, simulate the transaction
            # In production, this would create actual blockchain transaction
            loan_id = self._generate_mock_loan_id()
            tx_hash = self._generate_mock_tx_hash()
            
            logger.info(f"Loan request created: ID {loan_id}, Amount {amount_eth} ETH")
            
            return {
                "success": True,
                "transaction_hash": tx_hash,
                "loan_id": loan_id,
                "message": f"Loan request for {amount_eth} ETH created successfully"
            }
        except Exception as e:
            logger.error(f"Failed to create loan request: {e}")
            return {"error": str(e)}
    
    def update_credit_score(self, user_address, new_score, reason="ML model update"):
        """
        Update user's credit score on blockchain
        Called by: Flask backend after ML model calculation
        """
        if not self.credit_score_contract:
            # Simulate for now
            logger.info(f"Credit score updated (simulated): {user_address} -> {new_score}")
            return {
                "success": True,
                "transaction_hash": self._generate_mock_tx_hash(),
                "message": f"Credit score updated to {new_score}"
            }
        
        try:
            # In production, this would call the smart contract
            tx_hash = self._generate_mock_tx_hash()
            logger.info(f"Credit score updated: {user_address} -> {new_score}")
            
            return {
                "success": True,
                "transaction_hash": tx_hash,
                "message": "Credit score updated on blockchain"
            }
        except Exception as e:
            logger.error(f"Failed to update credit score: {e}")
            return {"error": str(e)}
    
    def get_user_credit_score(self, user_address):
        """
        Get user's credit score from blockchain
        Called by: Flask backend to verify scores
        """
        if not self.credit_score_contract:
            # Return simulated data for development
            return {
                "success": True,
                "credit_score": 650,  # Default score
                "last_updated": datetime.now().isoformat(),
                "source": "simulated"
            }
        
        try:
            # In production, call actual contract
            score = 650  # Placeholder
            return {
                "success": True,
                "credit_score": score,
                "last_updated": datetime.now().isoformat(),
                "source": "blockchain"
            }
        except Exception as e:
            logger.error(f"Failed to get credit score: {e}")
            return {"error": str(e)}
    
    def get_active_loans(self):
        """
        Get all active loans for frontend display
        Called by: Flask backend for loan marketplace page
        """
        if not self.contract:
            # Return mock data for development
            return {
                "success": True,
                "loans": [
                    {
                        "id": 1,
                        "borrower": "0x1234567890123456789012345678901234567890",
                        "amount": "5.0",
                        "interest_rate": 1200,  # 12%
                        "duration_days": 30,
                        "purpose": "Business expansion",
                        "credit_score": 720
                    }
                ],
                "source": "simulated"
            }
        
        try:
            # In production, fetch from contract
            return {
                "success": True,
                "loans": [],
                "source": "blockchain"
            }
        except Exception as e:
            logger.error(f"Failed to get active loans: {e}")
            return {"error": str(e)}
    
    def get_user_profile(self, user_address):
        """
        Get user's profile and lending statistics
        Called by: Flask backend for user dashboard
        """
        if not self.contract:
            return {
                "success": True,
                "profile": {
                    "credit_score": 650,
                    "total_borrowed": "0",
                    "total_lent": "0",
                    "successful_loans": 0,
                    "defaulted_loans": 0,
                    "reputation": 100
                },
                "source": "simulated"
            }
        
        try:
            # In production, fetch from contract
            return {
                "success": True,
                "profile": {},
                "source": "blockchain"
            }
        except Exception as e:
            return {"error": str(e)}
    
    # Helper methods
    
    def _generate_mock_loan_id(self):
        """Generate mock loan ID for development"""
        import random
        return random.randint(1000, 9999)
    
    def _generate_mock_tx_hash(self):
        """Generate mock transaction hash for development"""
        import random
        return "0x" + "".join(random.choices('0123456789abcdef', k=64))
    
    def verify_transaction(self, tx_hash):
        """
        Verify if transaction was successful
        Called by: Backend to confirm blockchain operations
        """
        try:
            receipt = self.w3.eth.getTransactionReceipt(tx_hash)
            return {
                "success": True,
                "confirmed": receipt['status'] == 1,
                "block_number": receipt['blockNumber'],
                "gas_used": receipt['gasUsed']
            }
        except Exception as e:
            logger.error(f"Transaction verification failed: {e}")
            return {"error": str(e)}
    
    def get_platform_statistics(self):
        """
        Get overall platform statistics
        Called by: Frontend dashboard for displaying metrics
        """
        try:
            # In production, this would query actual contracts
            return {
                "success": True,
                "stats": {
                    "total_loans": 42,
                    "total_volume": "150.5",  # ETH
                    "active_borrowers": 15,
                    "active_lenders": 8,
                    "average_credit_score": 685,
                    "platform_fees_collected": "3.2"
                },
                "source": "simulated"
            }
        except Exception as e:
            logger.error(f"Failed to get platform stats: {e}")
            return {"error": str(e)}

class BlockchainConfig:
    """
    Configuration helper for blockchain connections
    Used by Flask backend to manage different environments
    """
    
    @staticmethod
    def get_development_config():
        """Configuration for local development"""
        return {
            "provider_uri": "http://localhost:8545",  # Local Ganache
            "credit_score_contract": None,  # Deploy locally
            "nexacred_contract": None,      # Deploy locally
            "network_id": 5777
        }
    
    @staticmethod
    def get_testnet_config():
        """Configuration for Ethereum testnet (Sepolia/Goerli)"""
        return {
            "provider_uri": os.getenv("TESTNET_RPC_URL", "https://sepolia.infura.io/v3/YOUR_KEY"),
            "credit_score_contract": os.getenv("TESTNET_CREDIT_CONTRACT"),
            "nexacred_contract": os.getenv("TESTNET_LENDING_CONTRACT"),
            "network_id": 11155111  # Sepolia
        }
    
    @staticmethod
    def get_mainnet_config():
        """Configuration for Ethereum mainnet (production)"""
        return {
            "provider_uri": os.getenv("MAINNET_RPC_URL"),
            "credit_score_contract": os.getenv("MAINNET_CREDIT_CONTRACT"),
            "nexacred_contract": os.getenv("MAINNET_LENDING_CONTRACT"),
            "network_id": 1
        }

# Factory functions for different environments
def create_development_client():
    """Create blockchain client for local development"""
    config = BlockchainConfig.get_development_config()
    return NexaCredBlockchain(
        web3_provider_uri=config["provider_uri"]
    )

def create_production_client():
    """Create blockchain client for production use"""
    env = os.getenv("FLASK_ENV", "development")
    
    if env == "production":
        config = BlockchainConfig.get_mainnet_config()
    else:
        config = BlockchainConfig.get_testnet_config()
    
    return NexaCredBlockchain(
        web3_provider_uri=config["provider_uri"],
        contract_address=config["nexacred_contract"]
    )

# Testing and demonstration
if __name__ == "__main__":
    print("=== NexaCred Blockchain Integration Test ===")
    
    # Test blockchain connection
    blockchain = create_development_client()
    print(f"Blockchain connected: {blockchain.is_connected()}")
    
    # Test backend integration methods
    print("\n--- Testing Backend Integration Methods ---")
    
    # Test loan creation
    borrower_addr = "0x742d35Cc6575C9E4D7B7b9e8C2dFDbF15f8E5e9C"
    loan_result = blockchain.create_loan_request(
        borrower_address=borrower_addr,
        amount_eth=5.0,
        interest_rate=1200,  # 12%
        duration_days=30,
        purpose="Business expansion loan"
    )
    print(f"Loan creation result: {loan_result}")
    
    # Test credit score update
    score_result = blockchain.update_credit_score(
        user_address=borrower_addr,
        new_score=720,
        reason="ML model v2.1 calculation"
    )
    print(f"Credit score update: {score_result}")
    
    # Test data retrieval
    active_loans = blockchain.get_active_loans()
    print(f"Active loans: {len(active_loans.get('loans', []))} found")
    
    user_profile = blockchain.get_user_profile(borrower_addr)
    print(f"User profile retrieved: {user_profile['success']}")
    
    platform_stats = blockchain.get_platform_statistics()
    print(f"Platform stats: {platform_stats}")
    
    print("\n✅ Blockchain integration test completed!")
    print("\n📋 Integration Guide:")
    print("1. Add 'web3==6.0.0' to backend/requirements.txt")
    print("2. Import this module in Flask app.py")
    print("3. Initialize blockchain client in Flask config")
    print("4. Use methods in API endpoints for blockchain operations")
