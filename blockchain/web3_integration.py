import json
import os
from web3 import Web3
from web3.middleware import geth_poa_middleware
from datetime import datetime
import requests

class NexaCredBlockchain:
    def __init__(self, web3_provider_uri=None, contract_address=None, private_key=None):
        # Initialize Web3 connection
        if web3_provider_uri:
            if web3_provider_uri.startswith('http'):
                self.w3 = Web3(Web3.HTTPProvider(web3_provider_uri))
            else:
                self.w3 = Web3(Web3.IPCProvider(web3_provider_uri))
        else:
            # Default to local Ganache
            self.w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))
        
        # Add PoA middleware for compatibility
        self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        self.private_key = private_key
        self.contract_address = contract_address
        self.contract = None
        
        # Contract ABI (simplified for demo)
        self.contract_abi = [
            {
                "inputs": [],
                "name": "getContractStats",
                "outputs": [
                    {"internalType": "uint256", "name": "totalLoans", "type": "uint256"},
                    {"internalType": "uint256", "name": "totalOffers", "type": "uint256"},
                    {"internalType": "uint256", "name": "totalValueLocked", "type": "uint256"}
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "uint256", "name": "amount", "type": "uint256"},
                    {"internalType": "uint256", "name": "interestRate", "type": "uint256"},
                    {"internalType": "uint256", "name": "duration", "type": "uint256"},
                    {"internalType": "uint256", "name": "creditScore", "type": "uint256"},
                    {"internalType": "uint256", "name": "riskScore", "type": "uint256"},
                    {"internalType": "string", "name": "purpose", "type": "string"}
                ],
                "name": "requestLoan",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "uint256", "name": "loanId", "type": "uint256"}
                ],
                "name": "fundLoan",
                "outputs": [],
                "stateMutability": "payable",
                "type": "function"
            },
            {
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "internalType": "uint256", "name": "loanId", "type": "uint256"},
                    {"indexed": True, "internalType": "address", "name": "borrower", "type": "address"},
                    {"indexed": False, "internalType": "uint256", "name": "amount", "type": "uint256"},
                    {"indexed": False, "internalType": "uint256", "name": "interestRate", "type": "uint256"},
                    {"indexed": False, "internalType": "uint256", "name": "duration", "type": "uint256"},
                    {"indexed": False, "internalType": "string", "name": "purpose", "type": "string"}
                ],
                "name": "LoanRequested",
                "type": "event"
            }
        ]
        
        if contract_address:
            self.contract = self.w3.eth.contract(
                address=Web3.toChecksumAddress(contract_address),
                abi=self.contract_abi
            )
    
    def is_connected(self):
        """Check if Web3 is connected"""
        return self.w3.isConnected()
    
    def get_account(self, private_key=None):
        """Get account from private key"""
        if private_key:
            return self.w3.eth.account.privateKeyToAccount(private_key)
        elif self.private_key:
            return self.w3.eth.account.privateKeyToAccount(self.private_key)
        else:
            return None
    
    def get_balance(self, address):
        """Get ETH balance of an address"""
        return self.w3.eth.get_balance(Web3.toChecksumAddress(address))
    
    def request_loan(self, borrower_private_key, amount, interest_rate, duration, credit_score, risk_score, purpose):
        """Request a loan on the blockchain"""
        if not self.contract:
            return {"error": "Contract not initialized"}
        
        try:
            account = self.w3.eth.account.privateKeyToAccount(borrower_private_key)
            
            # Convert amount to Wei
            amount_wei = self.w3.toWei(amount, 'ether')
            
            # Build transaction
            transaction = self.contract.functions.requestLoan(
                amount_wei,
                interest_rate,
                duration,
                credit_score,
                risk_score,
                purpose
            ).buildTransaction({
                'from': account.address,
                'gas': 300000,
                'gasPrice': self.w3.toWei('20', 'gwei'),
                'nonce': self.w3.eth.getTransactionCount(account.address)
            })
            
            # Sign and send transaction
            signed_txn = self.w3.eth.account.signTransaction(transaction, borrower_private_key)
            tx_hash = self.w3.eth.sendRawTransaction(signed_txn.rawTransaction)
            
            return {
                "success": True,
                "transaction_hash": tx_hash.hex(),
                "message": "Loan request submitted to blockchain"
            }
            
        except Exception as e:
            return {"error": f"Blockchain transaction failed: {str(e)}"}
    
    def fund_loan(self, lender_private_key, loan_id, amount):
        """Fund a loan on the blockchain"""
        if not self.contract:
            return {"error": "Contract not initialized"}
        
        try:
            account = self.w3.eth.account.privateKeyToAccount(lender_private_key)
            amount_wei = self.w3.toWei(amount, 'ether')
            
            transaction = self.contract.functions.fundLoan(loan_id).buildTransaction({
                'from': account.address,
                'value': amount_wei,
                'gas': 300000,
                'gasPrice': self.w3.toWei('20', 'gwei'),
                'nonce': self.w3.eth.getTransactionCount(account.address)
            })
            
            signed_txn = self.w3.eth.account.signTransaction(transaction, lender_private_key)
            tx_hash = self.w3.eth.sendRawTransaction(signed_txn.rawTransaction)
            
            return {
                "success": True,
                "transaction_hash": tx_hash.hex(),
                "message": "Loan funded on blockchain"
            }
            
        except Exception as e:
            return {"error": f"Blockchain transaction failed: {str(e)}"}
    
    def get_contract_stats(self):
        """Get contract statistics"""
        if not self.contract:
            return {"error": "Contract not initialized"}
        
        try:
            stats = self.contract.functions.getContractStats().call()
            return {
                "total_loans": stats[0],
                "total_offers": stats[1],
                "total_value_locked": self.w3.fromWei(stats[2], 'ether')
            }
        except Exception as e:
            return {"error": f"Failed to get contract stats: {str(e)}"}
    
    def get_transaction_receipt(self, tx_hash):
        """Get transaction receipt"""
        try:
            return self.w3.eth.getTransactionReceipt(tx_hash)
        except Exception as e:
            return {"error": f"Failed to get transaction receipt: {str(e)}"}
    
    def simulate_blockchain_transaction(self, tx_type, amount, from_address, to_address=None):
        """Simulate a blockchain transaction for demo purposes"""
        import random
        import time
        
        # Simulate transaction processing time
        time.sleep(1)
        
        tx_hash = f"0x{''.join(random.choices('0123456789abcdef', k=64))}"
        block_number = random.randint(1000000, 2000000)
        gas_used = random.randint(21000, 100000)
        
        transaction_data = {
            "hash": tx_hash,
            "type": tx_type,
            "amount": amount,
            "from": from_address,
            "to": to_address or "0x" + "".join(random.choices('0123456789abcdef', k=40)),
            "block_number": block_number,
            "gas_used": gas_used,
            "gas_price": 20000000000,  # 20 Gwei
            "status": "confirmed",
            "timestamp": datetime.utcnow().isoformat(),
            "confirmations": random.randint(1, 20)
        }
        
        return transaction_data
    
    def deploy_contract(self, deployer_private_key):
        """Deploy the NexaCred contract (mock implementation)"""
        account = self.w3.eth.account.privateKeyToAccount(deployer_private_key)
        
        # Mock deployment - in reality, you'd compile and deploy the Solidity contract
        mock_contract_address = "0x" + "".join(['1234567890abcdef'[i % 16] for i in range(40)])
        
        deployment_data = {
            "contract_address": mock_contract_address,
            "deployer": account.address,
            "transaction_hash": f"0x{''.join(['0123456789abcdef'[i % 16] for i in range(64)])}",
            "block_number": 1000001,
            "gas_used": 2500000,
            "deployment_timestamp": datetime.utcnow().isoformat()
        }
        
        # Update contract address
        self.contract_address = mock_contract_address
        self.contract = self.w3.eth.contract(
            address=Web3.toChecksumAddress(mock_contract_address),
            abi=self.contract_abi
        )
        
        return deployment_data

# Factory function for easy instantiation
def create_blockchain_client(provider_uri=None, contract_address=None, private_key=None):
    return NexaCredBlockchain(provider_uri, contract_address, private_key)

# Example usage and testing
if __name__ == "__main__":
    # Initialize blockchain client
    blockchain = create_blockchain_client()
    
    print(f"Web3 connected: {blockchain.is_connected()}")
    
    # Simulate some transactions
    print("\n--- Simulating Blockchain Transactions ---")
    
    # Mock addresses
    borrower_address = "0x1234567890123456789012345678901234567890"
    lender_address = "0x0987654321098765432109876543210987654321"
    
    # Simulate loan request
    loan_tx = blockchain.simulate_blockchain_transaction(
        "loan_request", 5.0, borrower_address
    )
    print(f"Loan Request Transaction: {loan_tx['hash']}")
    
    # Simulate loan funding
    funding_tx = blockchain.simulate_blockchain_transaction(
        "loan_funding", 5.0, lender_address, borrower_address
    )
    print(f"Loan Funding Transaction: {funding_tx['hash']}")
    
    # Simulate repayment
    repayment_tx = blockchain.simulate_blockchain_transaction(
        "loan_repayment", 5.5, borrower_address, lender_address
    )
    print(f"Loan Repayment Transaction: {repayment_tx['hash']}")
    
    print("\nBlockchain integration ready!")
