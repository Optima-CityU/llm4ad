#!/usr/bin/env python3
"""
Simple test runner for CO-Bench data loading tests.

Usage:
    python run_cobench_tests.py                    # Run all tests
    python run_cobench_tests.py --quick            # Run only quick tests (instantiation + dictionary structure)
    python run_cobench_tests.py --integration      # Run integration test only
"""

import sys
import unittest
import argparse


def run_all_tests():
    """Run all CO-Bench tests."""
    print("Running all CO-Bench data loading tests...")
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName('llm4ad.test.test_cobench_loading')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def run_quick_tests():
    """Run only the quick tests (instantiation and dictionary structure)."""
    print("Running quick CO-Bench tests...")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add specific quick test methods
    from llm4ad.test.test_cobench_loading import TestCOBenchDataLoading
    suite.addTest(TestCOBenchDataLoading('test_all_tasks_instantiate'))
    suite.addTest(TestCOBenchDataLoading('test_datasets_are_dictionaries'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def run_integration_tests():
    """Run only the integration tests."""
    print("Running CO-Bench integration tests...")
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName('llm4ad.test.test_cobench_loading.TestCOBenchIntegration')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description='Run CO-Bench data loading tests')
    parser.add_argument('--quick', action='store_true', 
                       help='Run only quick tests (instantiation + dictionary structure)')
    parser.add_argument('--integration', action='store_true',
                       help='Run only integration tests')
    
    args = parser.parse_args()
    
    success = False
    
    if args.quick:
        success = run_quick_tests()
    elif args.integration:
        success = run_integration_tests()
    else:
        success = run_all_tests()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == '__main__':
    main() 