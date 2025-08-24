#!/usr/bin/env python3
"""
NexaCred Complete System Test
============================

Comprehensive test suite for the cleaned up NexaCred platform
Tests all ML components, backend integration, and AI functionality
"""

import sys
import os
import requests
import json
import time
import subprocess
from datetime import datetime
from typing import Dict, Any, List

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'ml'))
sys.path.append(os.path.join(project_root, 'backend'))

class NexaCredSystemTester:
    """Comprehensive system tester for NexaCred platform"""

    def __init__(self):
        self.backend_url = "http://localhost:5000"
        self.test_results = {}
        self.backend_process = None

    def run_complete_test_suite(self):
        """Run the complete test suite"""
        print("🚀 NEXACRED COMPLETE SYSTEM TEST")
        print("=" * 60)
        print(f"Started at: {datetime.now()}")
        print()

        try:
            # Test 1: ML Components
            self.test_ml_components()

            # Test 2: Backend API
            self.test_backend_api()

            # Test 3: AI Integration
            self.test_ai_integration()

            # Test 4: End-to-End Workflow
            self.test_end_to_end_workflow()

            # Generate final report
            self.generate_test_report()

        except Exception as e:
            print(f"❌ Test suite failed: {e}")
            import traceback
            traceback.print_exc()

    def test_ml_components(self):
        """Test all ML components"""
        print("🧠 TESTING ML COMPONENTS")
        print("-" * 40)

        results = {}

        try:
            # Test 1: Traditional Credit Model
            print("1. Testing Traditional Credit Model...")
            from train_model import CreditScoreModel

            model = CreditScoreModel(random_state=42)
            X, y = model.generate_synthetic_data(n_samples=100, n_features=15)
            accuracy = model.train_model(X, y)

            # Test predictions
            test_data = {
                'annual_income': 750000,
                'debt_to_income_ratio': 0.2,
                'payment_history_score': 1.5,
                'credit_utilization_ratio': 0.15,
                'age': 30,
                'employment_tenure_months': 36
            }

            test_df = pd.DataFrame([test_data])
            category = model.predict_credit_score_category(test_df)[0]
            score = model.convert_category_to_score([category])[0]

            results['traditional_ml'] = {
                'status': 'PASS',
                'accuracy': accuracy,
                'test_score': score,
                'test_category': category
            }
            print(f"   ✅ Accuracy: {accuracy:.4f}")
            print(f"   ✅ Test score: {score}")

        except Exception as e:
            results['traditional_ml'] = {'status': 'FAIL', 'error': str(e)}
            print(f"   ❌ Failed: {e}")

        try:
            # Test 2: Granite AI System
            print("\n2. Testing IBM Granite AI System...")
            from granite_agents import IBMGraniteFinancialAI

            granite_ai = IBMGraniteFinancialAI()

            # Test credit analysis
            credit_result = granite_ai.analyze_credit_risk(test_data)

            # Test loan recommendation
            loan_result = granite_ai.generate_loan_recommendation(test_data, "personal")

            # Test fraud detection
            fraud_data = {
                'amount': 15000,
                'average_amount': 3000,
                'transactions_today': 5,
                'transaction_hour': 14,
                'new_location': False
            }
            fraud_result = granite_ai.detect_fraud(fraud_data)

            results['granite_ai'] = {
                'status': 'PASS',
                'model_loaded': granite_ai.model_loaded,
                'credit_score': credit_result.credit_score,
                'loan_eligible': loan_result.get('eligible'),
                'fraud_probability': fraud_result.get('fraud_probability')
            }
            print(f"   ✅ Model loaded: {granite_ai.model_loaded}")
            print(f"   ✅ Credit score: {credit_result.credit_score}")
            print(f"   ✅ Loan eligible: {loan_result.get('eligible')}")
            print(f"   ✅ Fraud probability: {fraud_result.get('fraud_probability'):.2%}")

        except Exception as e:
            results['granite_ai'] = {'status': 'FAIL', 'error': str(e)}
            print(f"   ❌ Failed: {e}")

        try:
            # Test 3: Financial Assistant
            print("\n3. Testing Financial Assistant...")
            from financial_assistant import NexaCredFinancialAssistant

            assistant = NexaCredFinancialAssistant()

            # Test credit scoring
            score_result = assistant.get_score("TEST_USER_001", test_data)

            # Test loan generation
            offer_result = assistant.generate_offer("TEST_USER_001", test_data, "personal")

            # Test fraud detection
            fraud_detection = assistant.detect_fraud(fraud_data)

            # Get system status
            system_status = assistant.get_system_status()

            results['financial_assistant'] = {
                'status': 'PASS',
                'primary_ai': system_status['primary_ai'],
                'granite_available': system_status['granite_available'],
                'credit_score': score_result['credit_score'],
                'loan_eligible': offer_result.get('eligible'),
                'system_health': system_status['system_health']
            }
            print(f"   ✅ Primary AI: {system_status['primary_ai']}")
            print(f"   ✅ Granite available: {system_status['granite_available']}")
            print(f"   ✅ Credit score: {score_result['credit_score']}")
            print(f"   ✅ System health: {system_status['system_health']}")

        except Exception as e:
            results['financial_assistant'] = {'status': 'FAIL', 'error': str(e)}
            print(f"   ❌ Failed: {e}")

        self.test_results['ml_components'] = results
        print()

    def test_backend_api(self):
        """Test backend API functionality"""
        print("🔗 TESTING BACKEND API")
        print("-" * 40)

        results = {}

        try:
            # Check if backend is running
            print("1. Checking backend connectivity...")
            try:
                response = requests.get(f"{self.backend_url}/", timeout=5)
                if response.status_code == 200:
                    print("   ✅ Backend is running")
                    backend_running = True
                else:
                    print(f"   ⚠️ Backend returned status {response.status_code}")
                    backend_running = False
            except requests.exceptions.RequestException:
                print("   ❌ Backend not accessible, starting it...")
                backend_running = False
                # Note: In production, you'd start the backend here

            if backend_running:
                # Test ML status endpoint
                print("\n2. Testing ML status endpoint...")
                try:
                    response = requests.get(f"{self.backend_url}/api/ml/status")
                    if response.status_code == 200:
                        ml_status = response.json()
                        print(f"   ✅ ML Status: {ml_status.get('ml_available')}")
                        results['ml_status'] = {'status': 'PASS', 'data': ml_status}
                    else:
                        print(f"   ❌ ML status endpoint failed: {response.status_code}")
                        results['ml_status'] = {'status': 'FAIL', 'error': f"HTTP {response.status_code}"}
                except Exception as e:
                    print(f"   ❌ ML status test failed: {e}")
                    results['ml_status'] = {'status': 'FAIL', 'error': str(e)}

                # Test credit scoring endpoint
                print("\n3. Testing credit scoring endpoint...")
                try:
                    test_payload = {
                        'user_id': 'TEST_USER_001',
                        'customer_data': {
                            'annual_income': 750000,
                            'debt_to_income_ratio': 0.2,
                            'payment_history_score': 1.5,
                            'credit_utilization_ratio': 0.15,
                            'age': 30,
                            'employment_tenure_months': 36
                        }
                    }

                    response = requests.post(
                        f"{self.backend_url}/api/credit-score",
                        json=test_payload,
                        headers={'Content-Type': 'application/json'}
                    )

                    if response.status_code == 200:
                        credit_data = response.json()
                        print(f"   ✅ Credit score: {credit_data.get('credit_score')}")
                        print(f"   ✅ Risk level: {credit_data.get('risk_level')}")
                        results['credit_scoring'] = {'status': 'PASS', 'data': credit_data}
                    else:
                        print(f"   ❌ Credit scoring failed: {response.status_code}")
                        results['credit_scoring'] = {'status': 'FAIL', 'error': f"HTTP {response.status_code}"}

                except Exception as e:
                    print(f"   ❌ Credit scoring test failed: {e}")
                    results['credit_scoring'] = {'status': 'FAIL', 'error': str(e)}
            else:
                results['backend_connectivity'] = {'status': 'FAIL', 'error': 'Backend not running'}

        except Exception as e:
            results['backend_api'] = {'status': 'FAIL', 'error': str(e)}
            print(f"❌ Backend API test failed: {e}")

        self.test_results['backend_api'] = results
        print()

    def test_ai_integration(self):
        """Test AI system integration"""
        print("🤖 TESTING AI INTEGRATION")
        print("-" * 40)

        results = {}

        try:
            print("1. Testing AI component compatibility...")

            # Import check
            import_success = True
            components = ['train_model', 'granite_agents', 'financial_assistant']

            for component in components:
                try:
                    __import__(component)
                    print(f"   ✅ {component} imported successfully")
                except ImportError as e:
                    print(f"   ❌ {component} import failed: {e}")
                    import_success = False

            results['import_compatibility'] = {'status': 'PASS' if import_success else 'FAIL'}

            print("\n2. Testing AI system interoperability...")

            # Cross-system test
            from financial_assistant import NexaCredFinancialAssistant
            from granite_agents import IBMGraniteFinancialAI

            assistant = NexaCredFinancialAssistant()
            granite = IBMGraniteFinancialAI()

            test_customer = {
                'annual_income': 850000,
                'debt_to_income_ratio': 0.25,
                'payment_history_score': 1.8,
                'credit_utilization_ratio': 0.15,
                'age': 32
            }

            # Compare results from different systems
            assistant_result = assistant.get_score("TEST_USER", test_customer)
            granite_result = granite.analyze_credit_risk(test_customer)

            score_diff = abs(assistant_result['credit_score'] - granite_result.credit_score)

            print(f"   ✅ Assistant score: {assistant_result['credit_score']}")
            print(f"   ✅ Granite score: {granite_result.credit_score}")
            print(f"   ✅ Score difference: {score_diff}")

            results['interoperability'] = {
                'status': 'PASS',
                'assistant_score': assistant_result['credit_score'],
                'granite_score': granite_result.credit_score,
                'score_difference': score_diff
            }

        except Exception as e:
            results['ai_integration'] = {'status': 'FAIL', 'error': str(e)}
            print(f"   ❌ AI integration test failed: {e}")

        self.test_results['ai_integration'] = results
        print()

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        print("🔄 TESTING END-TO-END WORKFLOW")
        print("-" * 40)

        results = {}

        try:
            print("1. Simulating complete customer journey...")

            # Step 1: Customer data input
            customer_profile = {
                'name': 'Test Customer',
                'annual_income': 750000,
                'debt_to_income_ratio': 0.22,
                'payment_history_score': 1.6,
                'credit_utilization_ratio': 0.18,
                'age': 28,
                'employment_tenure_months': 42,
                'number_of_late_payments': 1,
                'total_credit_limit': 150000,
                'current_debt_amount': 27000
            }

            print(f"   📋 Customer: {customer_profile['name']}")
            print(f"   💰 Income: ₹{customer_profile['annual_income']:,}")

            # Step 2: Credit scoring
            from financial_assistant import NexaCredFinancialAssistant
            assistant = NexaCredFinancialAssistant()

            score_result = assistant.get_score("END_TO_END_TEST", customer_profile)
            print(f"   📊 Credit Score: {score_result['credit_score']}")
            print(f"   🎯 Category: {score_result['category']}")
            print(f"   ⚠️ Risk Level: {score_result['risk_level']}")

            # Step 3: Loan recommendation
            loan_offer = assistant.generate_offer("END_TO_END_TEST", customer_profile, "personal")
            if loan_offer.get('eligible'):
                print(f"   ✅ Loan Eligible: Yes")
                print(f"   💵 Amount: ₹{loan_offer['recommended_amount']:,}")
                print(f"   📈 Rate: {loan_offer['interest_rate']}%")
                print(f"   📅 EMI: ₹{loan_offer['estimated_emi']:,.2f}")
            else:
                print(f"   ❌ Loan Eligible: No")

            # Step 4: Fraud monitoring
            transaction = {
                'amount': 25000,
                'average_amount': 8000,
                'transactions_today': 3,
                'transaction_hour': 15,
                'new_location': False
            }

            fraud_check = assistant.detect_fraud(transaction)
            print(f"   🛡️ Fraud Risk: {fraud_check['fraud_probability']:.1%}")
            print(f"   🔍 Action: {fraud_check['recommended_action']}")

            # Step 5: System performance
            system_status = assistant.get_system_status()
            print(f"   🏥 System Health: {system_status['system_health']}")
            print(f"   🤖 Primary AI: {system_status['primary_ai']}")

            results['end_to_end'] = {
                'status': 'PASS',
                'credit_score': score_result['credit_score'],
                'loan_eligible': loan_offer.get('eligible', False),
                'fraud_risk': fraud_check['fraud_probability'],
                'system_health': system_status['system_health']
            }

        except Exception as e:
            results['end_to_end'] = {'status': 'FAIL', 'error': str(e)}
            print(f"   ❌ End-to-end test failed: {e}")

        self.test_results['end_to_end_workflow'] = results
        print()

    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("📋 TEST REPORT")
        print("=" * 60)

        total_tests = 0
        passed_tests = 0

        for category, tests in self.test_results.items():
            print(f"\n📂 {category.upper()}:")

            if isinstance(tests, dict):
                for test_name, result in tests.items():
                    total_tests += 1
                    status = result.get('status', 'UNKNOWN')

                    if status == 'PASS':
                        passed_tests += 1
                        print(f"   ✅ {test_name}: PASS")
                    else:
                        print(f"   ❌ {test_name}: FAIL")
                        if 'error' in result:
                            print(f"      Error: {result['error']}")

        print(f"\n" + "=" * 60)
        print(f"📊 SUMMARY")
        print(f"=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/max(1,total_tests)*100):.1f}%")

        if passed_tests == total_tests:
            print(f"\n🎉 ALL TESTS PASSED! NexaCred system is ready for production.")
        else:
            print(f"\n⚠️ Some tests failed. Please review the issues above.")

        print(f"\nCompleted at: {datetime.now()}")

# Main execution
if __name__ == "__main__":
    # Import pandas for the test
    try:
        import pandas as pd
    except ImportError:
        print("Installing pandas...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
        import pandas as pd

    # Run the complete test suite
    tester = NexaCredSystemTester()
    tester.run_complete_test_suite()
