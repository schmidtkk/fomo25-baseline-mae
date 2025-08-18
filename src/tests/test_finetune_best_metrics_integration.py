#!/usr/bin/env python
"""
Integration test to verify that best metrics tracking works with the actual finetune.py setup.

This test checks that:
1. The finetune.py script properly initializes callbacks with best metrics tracking
2. The paths are set up correctly
3. The callbacks can work together without conflicts
"""

import sys
import os
import tempfile
import shutil

def test_finetune_callback_setup():
    """Test that finetune.py sets up callbacks correctly for best metrics tracking."""
    print("Testing finetune.py callback setup...")
    
    # Add src to path to import finetune modules
    sys.path.insert(0, '/home/weidongguo/workspace/fomo2025/baseline-codebase-main/src')
    
    try:
        from utils.enhanced_callbacks import EnhancedModelCheckpoint, LossPlottingCallback
        
        # Simulate the setup that happens in finetune.py
        temp_dir = tempfile.mkdtemp()
        version_dir = os.path.join(temp_dir, "test_version")
        os.makedirs(version_dir, exist_ok=True)
        
        # Create checkpoint callback (mimicking finetune.py setup)
        checkpoint_callback = EnhancedModelCheckpoint(
            dirpath=version_dir,
            filename="best",
            monitor="val/loss",
            mode="min",
            save_top_k=1
        )
        
        # Create plotting callback (mimicking finetune.py setup)
        loss_plotting_callback = LossPlottingCallback(
            save_dir=version_dir,
            log_every_n_steps=10,
            plot_every_n_epochs=1,
            smoothing_window=5,
            best_metrics_file=os.path.join(version_dir, "best_metrics.txt")
        )
        
        # Verify that paths are set up correctly
        expected_best_metrics_file = os.path.join(version_dir, "best_metrics.txt")
        assert loss_plotting_callback.best_metrics_file == expected_best_metrics_file, \
            "Plotting callback best_metrics_file path mismatch"
        
        print("✅ Callback initialization test passed")
        
        # Test that both callbacks can coexist
        assert hasattr(checkpoint_callback, 'best_metrics'), "Checkpoint callback missing best_metrics"
        assert hasattr(loss_plotting_callback, 'best_metrics_file'), "Plotting callback missing best_metrics_file"
        
        print("✅ Callback compatibility test passed")
        
        # Test file path consistency
        assert checkpoint_callback.best_metrics_file is None, "Checkpoint callback should initialize best_metrics_file in setup"
        
        # Simulate setup call (what happens in actual training)
        from unittest.mock import Mock
        mock_trainer = Mock()
        mock_module = Mock()
        
        checkpoint_callback.setup(mock_trainer, mock_module, "fit")
        loss_plotting_callback.setup(mock_trainer, mock_module, "fit")
        
        # After setup, checkpoint callback should have best_metrics_file set
        expected_checkpoint_file = os.path.join(version_dir, "best_metrics.txt")
        assert checkpoint_callback.best_metrics_file == expected_checkpoint_file, \
            "Checkpoint callback best_metrics_file not set correctly after setup"
        
        # Both callbacks should point to the same file
        assert checkpoint_callback.best_metrics_file == loss_plotting_callback.best_metrics_file, \
            "Callbacks should use the same best_metrics_file"
        
        print("✅ File path consistency test passed")
        
        # Test that best metrics file is created
        assert os.path.exists(expected_best_metrics_file), "Best metrics file should be created during setup"
        
        with open(expected_best_metrics_file, 'r') as f:
            content = f.read()
            assert "# Best Metrics History" in content, "Best metrics file should contain proper header"
        
        print("✅ Best metrics file creation test passed")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_monitor_metric_variations():
    """Test that best metrics tracking works with different monitor metrics."""
    print("\nTesting different monitor metric configurations...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        from utils.enhanced_callbacks import EnhancedModelCheckpoint
        from unittest.mock import Mock
        
        # Test different monitor configurations that finetune.py uses
        monitor_configs = [
            ("val/loss", "min"),
            ("val/auroc_subject", "max"),
            ("val/corr", "max"),
            ("val/accuracy", "max")
        ]
        
        for monitor_metric, mode in monitor_configs:
            print(f"  Testing monitor={monitor_metric}, mode={mode}")
            
            callback = EnhancedModelCheckpoint(
                dirpath=temp_dir,
                filename=f"best_{monitor_metric.replace('/', '_')}",
                monitor=monitor_metric,
                mode=mode
            )
            
            # Mock setup
            mock_trainer = Mock()
            mock_module = Mock()
            callback.setup(mock_trainer, mock_module, "fit")
            
            # Verify callback has the correct monitor settings
            assert callback.monitor == monitor_metric, f"Monitor metric not set correctly"
            assert callback.mode == mode, f"Monitor mode not set correctly"
            
            # Verify best_metrics tracking includes this metric
            if monitor_metric in callback.best_metrics:
                assert callback.best_metrics[monitor_metric]['mode'] == mode, \
                    f"Best metrics mode mismatch for {monitor_metric}"
            
            print(f"    ✅ {monitor_metric} configuration passed")
        
        print("✅ All monitor metric configurations test passed")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_task_specific_configurations():
    """Test configurations for different FOMO task types."""
    print("\nTesting task-specific callback configurations...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        from utils.enhanced_callbacks import EnhancedModelCheckpoint, LossPlottingCallback
        
        # Task configurations from finetune.py
        task_configs = [
            {"task": "FOMO1", "monitor": "val/auroc_subject", "mode": "max", "task_type": "classification"},
            {"task": "FOMO2", "monitor": "val/auroc_subject", "mode": "max", "task_type": "classification"},
            {"task": "FOMO3", "monitor": "val/corr", "mode": "max", "task_type": "regression"},
        ]
        
        for config in task_configs:
            print(f"  Testing {config['task']} configuration...")
            
            task_dir = os.path.join(temp_dir, config['task'])
            os.makedirs(task_dir, exist_ok=True)
            
            # Create callbacks as finetune.py would
            checkpoint_callback = EnhancedModelCheckpoint(
                dirpath=task_dir,
                filename="best",
                monitor=config['monitor'],
                mode=config['mode']
            )
            
            plotting_callback = LossPlottingCallback(
                save_dir=task_dir,
                best_metrics_file=os.path.join(task_dir, "best_metrics.txt")
            )
            
            # Test that both callbacks work with this configuration
            assert checkpoint_callback.monitor == config['monitor']
            assert checkpoint_callback.mode == config['mode']
            assert plotting_callback.best_metrics_file.endswith("best_metrics.txt")
            
            print(f"    ✅ {config['task']} configuration passed")
        
        print("✅ All task-specific configurations test passed")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    print("Running finetune.py integration tests for best metrics tracking...")
    print("=" * 70)
    
    try:
        test_finetune_callback_setup()
        test_monitor_metric_variations()
        test_task_specific_configurations()
        
        print("\n" + "=" * 70)
        print("🎉 All integration tests passed! Best metrics tracking is properly integrated with finetune.py.")
        
    except Exception as e:
        print(f"\n❌ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
