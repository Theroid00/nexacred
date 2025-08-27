#!/usr/bin/env python3
"""
NexaCred Complete System Test - Updated for Optimized ML Structure
==================================================================

Comprehensive test suite for the optimized NexaCred platform
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
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, 'ml'))
sys.path.append(os.path.join(project_root, 'backend'))

class NexaCredSystemTester:
    """Comprehensive system tester for optimized NexaCred platform"""

    def __init__(self):
        self.backend_url = "http://localhost:5000"
        self.test_results = {}
        self.backend_process = None

    def run_complete_test_suite(self):
        """Run the complete test suite"""
        print("🚀 NEXACRED COMPLETE SYSTEM TEST - OPTIMIZED ML STRUCTURE")
        print("=" * 70)
        print(f"Started at: {datetime.now()}")
        print()

        try:
            # Test 1: ML Components
            self.test_ml_components()

            # Test 2: Backend Integration
            self.test_backend_integration()

            # Test 3: AI Integration
            self.test_ai_integration()

            # Test 4: End-to-End Workflow
            self.test_end_to_end_workflow()

            # Test 5: Performance Testing
            self.test_performance()

        except Exception as e:
            print(f"❌ Test suite failed: {e}")
        finally:
            self.cleanup()

        # Print final results
        self.print_test_summary()

    def test_ml_components(self):
        """Test core ML components"""
        print("🧪 TESTING ML COMPONENTS")
        print("-" * 40)

        # Test 1.1: Data Preprocessor
        try:
            from data_preprocessor import NexaCreditDataPreprocessor

            preprocessor = NexaCreditDataPreprocessor()
            print("✅ Data preprocessor imported successfully")

            # Test with sample data
            import pandas as pd
            import numpy as np

            sample_data = pd.DataFrame({
                'Age': [25, 35, 45, np.nan, 65],
                'Annual_Income': [500000, 800000, 1200000, 600000, 900000],
                'Credit_Utilization_Ratio': [0.3, 0.15, 0.45, 0.6, 0.2],
                'Debt_to_Income_Ratio': [0.4, 0.25, 0.5, 0.35, 0.3],
                'Num_of_Delayed_Payment': [0, 1, 3, 2, 0],
                'Credit_Score': ['Good', 'Good', 'Standard', 'Poor', 'Good']
            })

            sample_data.to_csv('temp_sample.csv', index=False)
            processed_data = preprocessor.preprocess_dataset('temp_sample.csv', is_training=True)
            os.remove('temp_sample.csv')

            print(f"✅ Data preprocessing works: {processed_data.shape}")
            self.test_results['data_preprocessor'] = 'PASS'

        except Exception as e:
            print(f"❌ Data preprocessor failed: {e}")
            self.test_results['data_preprocessor'] = 'FAIL'

        # Test 1.2: Hybrid Credit System
        try:
            from hybrid_credit_system import HybridCreditScoringSystem

            credit_system = HybridCreditScoringSystem()
            print("✅ Hybrid credit system imported successfully")
            self.test_results['hybrid_credit_system'] = 'PASS'

        except Exception as e:
            print(f"❌ Hybrid credit system failed: {e}")
            self.test_results['hybrid_credit_system'] = 'FAIL'

        # Test 1.3: Financial Assistant
        try:
            from financial_assistant import NexaCredFinancialAssistant

            assistant = NexaCredFinancialAssistant()

            # Test credit scoring
            test_data = {
                'annual_income': 800000,
                'debt_to_income_ratio': 0.25,
                'credit_utilization_ratio': 0.15,
                'number_of_late_payments': 0,
                'age': 35
            }

            result = assistant.get_score("test_user", test_data)
            print(f"✅ Financial assistant works: Score = {result.get('credit_score')}")

            # Test loan offer
            offer = assistant.generate_offer("test_user", test_data)
            print(f"✅ Loan offer generation works: Approved = {offer.get('offer', {}).get('approved')}")

            # Test fraud detection
            fraud_result = assistant.detect_fraud({
                'amount': 10000,
                'daily_transaction_count': 5,
                'location': 'verified',
                'hour': 14
            })
            print(f"✅ Fraud detection works: Risk = {fraud_result.get('fraud_likelihood')}")

            self.test_results['financial_assistant'] = 'PASS'

        except Exception as e:
            print(f"❌ Financial assistant failed: {e}")
            self.test_results['financial_assistant'] = 'FAIL'

        # Test 1.4: Granite AI Stub
        try:
            from granite_agents import IBMGraniteFinancialAI

            granite_ai = IBMGraniteFinancialAI()
            advice = granite_ai.generate_financial_advice("How to improve credit score?")
            print(f"✅ Granite AI works: {advice[:50]}...")

            profile = granite_ai.analyze_credit_profile(test_data)
            print(f"✅ Credit analysis works: Strength = {profile.get('profile_strength')}")

            self.test_results['granite_ai'] = 'PASS'

        except Exception as e:
            print(f"❌ Granite AI failed: {e}")
            self.test_results['granite_ai'] = 'FAIL'

        print()

    def test_backend_integration(self):
        """Test backend integration"""
        print("🔗 TESTING BACKEND INTEGRATION")
        print("-" * 40)

        try:
            # Start backend server
            backend_path = os.path.join(project_root, 'backend', 'app.py')

            if os.path.exists(backend_path):
                print("⏳ Starting backend server...")
                self.backend_process = subprocess.Popen(
                    [sys.executable, backend_path],
                    cwd=os.path.join(project_root, 'backend'),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                # Wait for server to start
                time.sleep(3)

                # Test health check
                try:
                    response = requests.get(f"{self.backend_url}/", timeout=5)
                    if response.status_code == 200:
                        print("✅ Backend server is running")
                        self.test_results['backend_startup'] = 'PASS'
                    else:
                        print(f"⚠️ Backend responded with status {response.status_code}")
                        self.test_results['backend_startup'] = 'PARTIAL'

                except requests.exceptions.RequestException:
                    print("❌ Backend server not responding")
                    self.test_results['backend_startup'] = 'FAIL'

            else:
                print("❌ Backend app.py not found")
                self.test_results['backend_startup'] = 'FAIL'

        except Exception as e:
            print(f"❌ Backend integration failed: {e}")
            self.test_results['backend_startup'] = 'FAIL'

        print()

    def test_ai_integration(self):
        """Test AI integration capabilities"""
        print("🧠 TESTING AI INTEGRATION")
        print("-" * 40)

        try:
            from financial_assistant import NexaCredFinancialAssistant
            from granite_agents import IBMGraniteFinancialAI

            assistant = NexaCredFinancialAssistant()
            granite_ai = IBMGraniteFinancialAI()

            # Test various customer profiles
            test_profiles = [
                {
                    'name': 'High Income Customer',
                    'data': {
                        'annual_income': 1500000,
                        'debt_to_income_ratio': 0.15,
                        'credit_utilization_ratio': 0.1,
                        'number_of_late_payments': 0,
                        'age': 40
                    }
                },
                {
                    'name': 'Average Customer',
                    'data': {
                        'annual_income': 600000,
                        'debt_to_income_ratio': 0.35,
                        'credit_utilization_ratio': 0.4,
                        'number_of_late_payments': 2,
                        'age': 30
                    }
                },
                {
                    'name': 'High Risk Customer',
                    'data': {
                        'annual_income': 300000,
                        'debt_to_income_ratio': 0.7,
                        'credit_utilization_ratio': 0.9,
                        'number_of_late_payments': 8,
                        'age': 25
                    }
                }
            ]

            for profile in test_profiles:
                print(f"🔍 Testing {profile['name']}:")

                # Credit scoring
                score_result = assistant.get_score(f"test_{profile['name'].lower().replace(' ', '_')}", profile['data'])
                print(f"   Score: {score_result.get('credit_score')} ({score_result.get('credit_category')})")

                # Loan offer
                offer_result = assistant.generate_offer(f"test_{profile['name'].lower().replace(' ', '_')}", profile['data'])
                if offer_result.get('offer', {}).get('approved'):
                    print(f"   Loan: Approved ₹{offer_result['offer']['max_amount']} at {offer_result['offer']['interest_rate']}%")
                else:
                    print(f"   Loan: Declined - {offer_result.get('offer', {}).get('reason', 'Unknown')}")

                # AI advice
                advice = granite_ai.analyze_credit_profile(profile['data'])
                print(f"   AI Analysis: {advice.get('profile_strength')} profile")
                print()

            self.test_results['ai_integration'] = 'PASS'
            print("✅ AI integration tests completed successfully")

        except Exception as e:
            print(f"❌ AI integration failed: {e}")
            self.test_results['ai_integration'] = 'FAIL'

        print()

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        print("🔄 TESTING END-TO-END WORKFLOW")
        print("-" * 40)

        try:
            # Simulate complete customer journey
            customer_data = {
                'annual_income': 750000,
                'debt_to_income_ratio': 0.28,
                'credit_utilization_ratio': 0.22,
                'number_of_late_payments': 1,
                'age': 32
            }

            print("1️⃣ Customer applies for credit assessment...")

            # Step 1: Credit scoring
            from financial_assistant import NexaCredFinancialAssistant
            assistant = NexaCredFinancialAssistant()

            score_result = assistant.get_score("e2e_customer", customer_data)
            print(f"   ✅ Credit Score: {score_result.get('credit_score')} ({score_result.get('credit_category')})")

            # Step 2: Generate loan offer
            print("2️⃣ Generating loan offer...")
            offer_result = assistant.generate_offer("e2e_customer", customer_data, "personal")

            if offer_result.get('offer', {}).get('approved'):
                print(f"   ✅ Approved: ₹{offer_result['offer']['max_amount']}")
                print(f"   Interest Rate: {offer_result['offer']['interest_rate']}%")
                print(f"   Term: {offer_result['offer']['term_months']} months")

                # Step 3: Fraud check on potential transaction
                print("3️⃣ Performing fraud check...")
                fraud_result = assistant.detect_fraud({
                    'amount': offer_result['offer']['max_amount'] * 0.1,  # 10% of approved amount
                    'daily_transaction_count': 1,
                    'location': 'verified',
                    'hour': 15,
                    'transaction_id': 'e2e_test_001'
                })

                print(f"   ✅ Fraud Risk: {fraud_result.get('fraud_likelihood')}")
                print(f"   Recommendation: {fraud_result.get('recommendation')}")

                # Step 4: AI advice
                print("4️⃣ Generating AI advice...")
                from granite_agents import IBMGraniteFinancialAI
                granite_ai = IBMGraniteFinancialAI()

                advice = granite_ai.generate_financial_advice("How can I maintain good credit after taking this loan?")
                print(f"   ✅ AI Advice: {advice[:100]}...")

                print("🎉 End-to-end workflow completed successfully!")
                self.test_results['e2e_workflow'] = 'PASS'

            else:
                print(f"   ❌ Loan declined: {offer_result.get('offer', {}).get('reason')}")
                self.test_results['e2e_workflow'] = 'PARTIAL'

        except Exception as e:
            print(f"❌ End-to-end workflow failed: {e}")
            self.test_results['e2e_workflow'] = 'FAIL'

        print()

    def test_performance(self):
        """Test system performance"""
        print("⚡ TESTING PERFORMANCE")
        print("-" * 40)

        try:
            from financial_assistant import NexaCredFinancialAssistant
            import time

            assistant = NexaCredFinancialAssistant()

            # Performance test data
            test_data = {
                'annual_income': 800000,
                'debt_to_income_ratio': 0.25,
                'credit_utilization_ratio': 0.15,
                'number_of_late_payments': 0,
                'age': 35
            }

            # Test credit scoring performance
            start_time = time.time()
            for i in range(100):
                assistant.get_score(f"perf_test_{i}", test_data)
            end_time = time.time()

            avg_time = (end_time - start_time) / 100
            print(f"✅ Credit scoring: {avg_time:.4f}s per request (100 requests)")

            # Test loan offer performance
            start_time = time.time()
            for i in range(50):
                assistant.generate_offer(f"perf_test_{i}", test_data)
            end_time = time.time()

            avg_time = (end_time - start_time) / 50
            print(f"✅ Loan offers: {avg_time:.4f}s per request (50 requests)")

            self.test_results['performance'] = 'PASS'
            print("✅ Performance tests completed")

        except Exception as e:
            print(f"❌ Performance testing failed: {e}")
            self.test_results['performance'] = 'FAIL'

        print()

    def cleanup(self):
        """Clean up test resources"""
        if self.backend_process:
            print("🧹 Cleaning up backend server...")
            self.backend_process.terminate()
            time.sleep(2)
            if self.backend_process.poll() is None:
                self.backend_process.kill()

        # Clean up any temp files
        temp_files = ['temp_sample.csv', 'temp_train_sample.csv', 'temp_test_sample.csv']
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def print_test_summary(self):
        """Print final test summary"""
        print("📊 TEST SUMMARY")
        print("=" * 50)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result == 'PASS')
        partial_tests = sum(1 for result in self.test_results.values() if result == 'PARTIAL')
        failed_tests = sum(1 for result in self.test_results.values() if result == 'FAIL')

        for test_name, result in self.test_results.items():
            status_emoji = {"PASS": "✅", "PARTIAL": "⚠️", "FAIL": "❌"}[result]
            print(f"{status_emoji} {test_name}: {result}")

        print()
        print(f"📈 Results: {passed_tests}/{total_tests} PASSED")
        if partial_tests > 0:
            print(f"⚠️ {partial_tests} tests had partial success")
        if failed_tests > 0:
            print(f"❌ {failed_tests} tests failed")

        if failed_tests == 0:
            print("🎉 ALL CRITICAL TESTS PASSED!")
        elif passed_tests >= total_tests * 0.7:
            print("✅ SYSTEM IS MOSTLY FUNCTIONAL")
        else:
            print("⚠️ SYSTEM NEEDS ATTENTION")

        print(f"Completed at: {datetime.now()}")
        print("=" * 50)

def main():
    """Run the complete system test"""
    tester = NexaCredSystemTester()
    tester.run_complete_test_suite()

if __name__ == "__main__":
    main()
