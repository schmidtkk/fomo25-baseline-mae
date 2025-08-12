#!/usr/bin/env python3
"""
Test runner for all enhanced features.
Runs comprehensive unit tests for the enhanced training system.
"""

import unittest
import sys
import os
from pathlib import Path

# Add src to path for imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

def run_enhanced_tests():
    """Run all enhanced feature tests"""
    
    # Test discovery
    test_dir = Path(__file__).parent
    loader = unittest.TestLoader()
    
    # Enhanced feature test modules
    enhanced_test_modules = [
        'test_subject_aggregation',
        'test_robust_early_stopping', 
        'test_training_visualizer',
        'test_enhanced_supervised_cls',
        'test_enhanced_finetune',
        'test_enhanced_analysis'
    ]
    
    # Create test suite
    suite = unittest.TestSuite()
    
    print("=" * 80)
    print("🧠 FOMO25 Enhanced Features Test Suite")
    print("=" * 80)
    print()
    
    # Load tests for each module
    for module_name in enhanced_test_modules:
        print(f"📋 Loading tests from {module_name}...")
        try:
            module_tests = loader.loadTestsFromName(module_name)
            suite.addTests(module_tests)
            test_count = module_tests.countTestCases()
            print(f"   ✅ Loaded {test_count} tests")
        except Exception as e:
            print(f"   ❌ Failed to load {module_name}: {e}")
    
    print()
    total_tests = suite.countTestCases()
    print(f"🎯 Total tests loaded: {total_tests}")
    print()
    
    # Run tests
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        failfast=False
    )
    
    print("🚀 Running enhanced features tests...")
    print("-" * 80)
    
    result = runner.run(suite)
    
    # Print summary
    print()
    print("=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED!")
        print(f"✅ Ran {result.testsRun} tests successfully")
    else:
        print("❌ SOME TESTS FAILED")
        print(f"📈 Tests run: {result.testsRun}")
        print(f"❌ Failures: {len(result.failures)}")
        print(f"💥 Errors: {len(result.errors)}")
        
        if result.failures:
            print("\n🔍 FAILURES:")
            for test, traceback in result.failures:
                print(f"  • {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
        
        if result.errors:
            print("\n💥 ERRORS:")
            for test, traceback in result.errors:
                print(f"  • {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
    
    print()
    print("=" * 80)
    
    # Return success status
    return result.wasSuccessful()


def run_specific_test_category(category):
    """Run tests for a specific category"""
    
    category_map = {
        'aggregation': ['test_subject_aggregation'],
        'early_stopping': ['test_robust_early_stopping'],
        'visualization': ['test_training_visualizer'],
        'model': ['test_enhanced_supervised_cls'],
        'finetune': ['test_enhanced_finetune'],
        'analysis': ['test_enhanced_analysis']
    }
    
    if category not in category_map:
        print(f"❌ Unknown category: {category}")
        print(f"Available categories: {list(category_map.keys())}")
        return False
    
    print(f"🎯 Running {category} tests...")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for module_name in category_map[category]:
        try:
            module_tests = loader.loadTestsFromName(module_name)
            suite.addTests(module_tests)
        except Exception as e:
            print(f"❌ Failed to load {module_name}: {e}")
            return False
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def main():
    """Main test runner function"""
    
    if len(sys.argv) > 1:
        # Run specific category
        category = sys.argv[1]
        success = run_specific_test_category(category)
    else:
        # Run all tests
        success = run_enhanced_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
