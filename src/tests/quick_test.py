#!/usr/bin/env python3
"""
Quick test runner script for individual enhanced feature components.
Use this for rapid testing during development.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add src to Python path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

def run_single_test(test_file):
    """Run a single test file"""
    print(f"🧪 Running {test_file}...")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'unittest', test_file, '-v'
        ], cwd=repo_root, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {test_file} passed!")
        else:
            print(f"❌ {test_file} failed!")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"💥 Error running {test_file}: {e}")
        return False

def main():
    """Quick test menu"""
    
    tests = {
        '1': ('Subject Aggregation', 'tests.test_subject_aggregation'),
        '2': ('Robust Early Stopping', 'tests.test_robust_early_stopping'),
        '3': ('Training Visualization', 'tests.test_training_visualizer'),
        '4': ('Enhanced Model', 'tests.test_enhanced_supervised_cls'),
        '5': ('Enhanced Finetune', 'tests.test_enhanced_finetune'),
        '6': ('Enhanced Analysis', 'tests.test_enhanced_analysis'),
        '7': ('All Enhanced Tests', 'all')
    }
    
    print("🧠 FOMO25 Enhanced Features - Quick Test Runner")
    print("=" * 60)
    
    for key, (name, _) in tests.items():
        print(f"{key}. {name}")
    
    print("\nEnter test number (or 'q' to quit): ", end='')
    choice = input().strip()
    
    if choice.lower() == 'q':
        return
    
    if choice not in tests:
        print("❌ Invalid choice!")
        return
    
    name, test_module = tests[choice]
    
    if test_module == 'all':
        # Run all tests
        print("\n🚀 Running all enhanced feature tests...")
        subprocess.run([sys.executable, 'tests/run_enhanced_tests.py'], cwd=repo_root)
    else:
        # Run specific test
        print(f"\n🧪 Running {name} tests...")
        run_single_test(test_module)

if __name__ == '__main__':
    main()
