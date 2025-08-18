#!/usr/bin/env python
"""
Simple validation test for best metrics tracking functionality.

This test can be run to verify that the enhanced callbacks work correctly
without needing complex mocking or dependencies.
"""

import os
import tempfile
import shutil

def test_best_metrics_file_creation():
    """Test basic best metrics file creation and format."""
    print("Testing best metrics file creation...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Import here to avoid import errors if modules aren't available
        from utils.enhanced_callbacks import EnhancedModelCheckpoint
        
        # Create callback instance
        callback = EnhancedModelCheckpoint(
            dirpath=temp_dir,
            filename="test",
            monitor="val/loss"
        )
        
        # Set up best metrics file manually (simulating setup)
        callback.best_metrics_file = os.path.join(temp_dir, "best_metrics.txt")
        
        # Test initialization
        callback._initialize_best_metrics_file()
        
        # Check file was created
        assert os.path.exists(callback.best_metrics_file), "Best metrics file was not created"
        
        # Check file content
        with open(callback.best_metrics_file, 'r') as f:
            content = f.read()
            assert "# Best Metrics History" in content, "File header missing"
            assert "# Updated:" in content, "Timestamp missing"
            
        print("✅ Best metrics file creation test passed")
        
        # Test file update with some metrics
        callback.best_metrics['val/loss']['value'] = 0.123
        callback.best_metrics['val/loss']['epoch'] = 10
        callback.best_metrics['val/loss']['step'] = 500
        
        callback._save_best_metrics_to_file()
        
        # Read file and check content
        with open(callback.best_metrics_file, 'r') as f:
            content = f.read()
            assert "Best val/loss: 0.123000" in content, "Best metric not saved correctly"
            assert "epoch 10" in content, "Epoch not saved correctly"
            
        print("✅ Best metrics file update test passed")
        
        # Test loading from file
        new_callback = EnhancedModelCheckpoint(dirpath=temp_dir, filename="test2", monitor="val/loss")
        new_callback.best_metrics_file = callback.best_metrics_file
        new_callback._load_best_metrics_from_file()
        
        assert abs(new_callback.best_metrics['val/loss']['value'] - 0.123) < 1e-6, "Metric value not loaded correctly"
        assert new_callback.best_metrics['val/loss']['epoch'] == 10, "Epoch not loaded correctly"
        
        print("✅ Best metrics file loading test passed")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_plotting_callback_best_metrics():
    """Test plotting callback best metrics loading."""
    print("\nTesting plotting callback best metrics loading...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a sample best metrics file
        best_metrics_file = os.path.join(temp_dir, "best_metrics.txt")
        with open(best_metrics_file, 'w') as f:
            f.write("# Best Metrics History\n")
            f.write("# Updated: 2025-08-17 14:30:45\n")
            f.write("Best val/loss: 0.234567 (epoch 45, step 1234)\n")
            f.write("Best val/auroc_subject: 0.876543 (epoch 67, step 2345)\n")
            f.write("Best val/corr: 0.654321 (epoch 89, step 3456)\n")
        
        # Import and test
        from utils.enhanced_callbacks import LossPlottingCallback
        
        callback = LossPlottingCallback(
            save_dir=temp_dir,
            best_metrics_file=best_metrics_file
        )
        
        # Test loading
        best_metrics = callback._load_best_metrics()
        
        assert 'val/loss' in best_metrics, "val/loss not found in loaded metrics"
        assert 'val/auroc_subject' in best_metrics, "val/auroc_subject not found"
        assert 'val/corr' in best_metrics, "val/corr not found"
        
        assert abs(best_metrics['val/loss']['value'] - 0.234567) < 1e-6, "val/loss value incorrect"
        assert best_metrics['val/loss']['epoch'] == 45, "val/loss epoch incorrect"
        
        assert abs(best_metrics['val/auroc_subject']['value'] - 0.876543) < 1e-6, "val/auroc_subject value incorrect"
        assert best_metrics['val/auroc_subject']['epoch'] == 67, "val/auroc_subject epoch incorrect"
        
        print("✅ Plotting callback best metrics loading test passed")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_malformed_file_handling():
    """Test graceful handling of malformed best metrics files."""
    print("\nTesting malformed file handling...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Create malformed file
        malformed_file = os.path.join(temp_dir, "malformed.txt")
        with open(malformed_file, 'w') as f:
            f.write("# Header\n")
            f.write("Random text\n")
            f.write("Best val/loss: not_a_number (epoch invalid)\n")
            f.write("Best incomplete line\n")
            f.write("Best val/auroc_subject: 0.5 (no epoch info)\n")
        
        from utils.enhanced_callbacks import LossPlottingCallback
        
        callback = LossPlottingCallback(
            save_dir=temp_dir,
            best_metrics_file=malformed_file
        )
        
        # Should not crash
        best_metrics = callback._load_best_metrics()
        assert isinstance(best_metrics, dict), "Should return dict even with malformed file"
        
        print("✅ Malformed file handling test passed")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    print("Running best metrics tracking validation tests...")
    print("=" * 60)
    
    try:
        test_best_metrics_file_creation()
        test_plotting_callback_best_metrics()
        test_malformed_file_handling()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! Best metrics tracking functionality is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise
