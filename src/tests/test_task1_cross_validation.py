import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock
import numpy as np

from utils.cross_validation import create_classification_stratified_folds
from utils.aggregate_kfold_task1_results import (
    extract_metrics_from_tensorboard_logs,
    aggregate_kfold_metrics,
    extract_fold_number
)


class TestTask1CrossValidation(unittest.TestCase):
    """
    Test suite for Task 1 cross-validation functionality.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_classification_stratified_folds(self):
        """Test creation of class-stratified folds for binary classification."""
        # Create mock subjects and labels with balanced classes
        subjects = [f"FOMO1_sub_{i:03d}" for i in range(20)]
        labels = [0, 1] * 10  # Alternating 0/1 for balanced dataset
        
        folds = create_classification_stratified_folds(subjects, labels, k=5)
        
        # Should have 5 folds
        self.assertEqual(len(folds), 5)
        
        # Each fold should have train/val split
        total_subjects_check = set()
        for fold_idx, (train_subjects, val_subjects) in enumerate(folds):
            # Basic sanity checks
            self.assertGreater(len(train_subjects), 0)
            self.assertGreater(len(val_subjects), 0)
            
            # No overlap between train and val
            self.assertEqual(len(set(train_subjects) & set(val_subjects)), 0)
            
            # Each subject appears in exactly one validation set
            total_subjects_check.update(val_subjects)
        
        # All subjects should appear exactly once in validation
        self.assertEqual(total_subjects_check, set(subjects))
    
    def test_classification_folds_with_imbalanced_data(self):
        """Test stratified folds with imbalanced class distribution."""
        # Create imbalanced dataset (80% class 0, 20% class 1)
        subjects = [f"FOMO1_sub_{i:03d}" for i in range(25)]
        labels = [0] * 20 + [1] * 5
        
        folds = create_classification_stratified_folds(subjects, labels, k=5)
        
        # Should still create 5 folds
        self.assertEqual(len(folds), 5)
        
        # Check that minority class appears in multiple folds
        folds_with_minority = 0
        for train_subjects, val_subjects in folds:
            val_labels = []
            for subject in val_subjects:
                subject_idx = subjects.index(subject)
                val_labels.append(labels[subject_idx])
            
            if 1 in val_labels:  # Minority class present
                folds_with_minority += 1
        
        # Minority class should appear in multiple folds (stratification working)
        self.assertGreaterEqual(folds_with_minority, 2)
    
    def test_extract_fold_number(self):
        """Test extraction of fold numbers from experiment directory paths."""
        test_cases = [
            ("./runs/Task001_FOMO1/unet_xl/fomo1_kfold_fold0/version_0", 0),
            ("./runs/Task001_FOMO1/unet_xl/fomo1_kfold_fold3/version_0", 3),
            ("./runs/Task001_FOMO1/unet_xl/experiment_fold10/version_0", 10),
            ("./runs/Task001_FOMO1/unet_xl/no_fold_here/version_0", 0),  # Fallback
        ]
        
        for path, expected_fold in test_cases:
            with self.subTest(path=path):
                fold_num = extract_fold_number(path)
                self.assertEqual(fold_num, expected_fold)
    
    def test_extract_metrics_from_best_metrics_file(self):
        """Test extraction of metrics from best_metrics.txt file."""
        # Create mock best_metrics.txt file
        experiment_dir = os.path.join(self.temp_dir, "version_0")
        checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        best_metrics_file = os.path.join(checkpoints_dir, "best_metrics.txt")
        with open(best_metrics_file, 'w') as f:
            f.write("# Best Metrics History\n")
            f.write("# Updated: 2025-08-26 10:00:00\n")
            f.write("\n")
            f.write("Best val/auroc_subject: 0.850000 (epoch 15, step 1500)\n")
            f.write("Best val/f1: 0.750000 (epoch 12, step 1200)\n")
            f.write("Best val/accuracy: 0.800000 (epoch 10, step 1000)\n")
            f.write("Best val/loss: 0.450000 (epoch 8, step 800)\n")
        
        target_metrics = ['val/auroc_subject', 'val/f1', 'val/accuracy', 'val/loss']
        metrics = extract_metrics_from_tensorboard_logs(experiment_dir, target_metrics)
        
        # Should extract all target metrics
        self.assertEqual(len(metrics), 4)
        self.assertAlmostEqual(metrics['val/auroc_subject'], 0.85)
        self.assertAlmostEqual(metrics['val/f1'], 0.75)
        self.assertAlmostEqual(metrics['val/accuracy'], 0.80)
        self.assertAlmostEqual(metrics['val/loss'], 0.45)
    
    def test_aggregate_kfold_metrics(self):
        """Test aggregation of metrics across multiple folds."""
        # Create mock experiment directories with metrics
        experiment_dirs = []
        
        # Create 5 mock folds with varying performance
        fold_metrics = [
            {'val/auroc_subject': 0.80, 'val/f1': 0.70, 'val/accuracy': 0.75},
            {'val/auroc_subject': 0.85, 'val/f1': 0.75, 'val/accuracy': 0.80},
            {'val/auroc_subject': 0.82, 'val/f1': 0.72, 'val/accuracy': 0.77},
            {'val/auroc_subject': 0.88, 'val/f1': 0.78, 'val/accuracy': 0.83},
            {'val/auroc_subject': 0.84, 'val/f1': 0.74, 'val/accuracy': 0.79},
        ]
        
        for fold_idx, metrics in enumerate(fold_metrics):
            fold_dir = os.path.join(self.temp_dir, f"fomo1_kfold_fold{fold_idx}", "version_0")
            checkpoints_dir = os.path.join(fold_dir, "checkpoints")
            os.makedirs(checkpoints_dir, exist_ok=True)
            
            # Create best_metrics.txt for this fold
            best_metrics_file = os.path.join(checkpoints_dir, "best_metrics.txt")
            with open(best_metrics_file, 'w') as f:
                f.write("# Best Metrics History\n")
                for metric_name, metric_value in metrics.items():
                    f.write(f"Best {metric_name}: {metric_value:.6f} (epoch 10, step 1000)\n")
            
            experiment_dirs.append(fold_dir)
        
        # Test aggregation
        target_metrics = ['val/auroc_subject', 'val/f1', 'val/accuracy']
        aggregated = aggregate_kfold_metrics(experiment_dirs, target_metrics)
        
        # Should have aggregated statistics for all metrics
        self.assertEqual(len(aggregated), 3)
        
        # Check AUROC statistics (mean should be close to manual calculation)
        auroc_stats = aggregated['val/auroc_subject']
        expected_auroc_mean = np.mean([0.80, 0.85, 0.82, 0.88, 0.84])
        self.assertAlmostEqual(auroc_stats['mean'], expected_auroc_mean, places=3)
        self.assertEqual(auroc_stats['n_folds'], 5)
        self.assertAlmostEqual(auroc_stats['min'], 0.80, places=2)
        self.assertAlmostEqual(auroc_stats['max'], 0.88, places=2)
    
    def test_cross_validation_edge_cases(self):
        """Test edge cases for cross-validation."""
        
        # Test with minimum subjects for 3-fold (need at least 3 samples per class)
        subjects = ["sub1", "sub2", "sub3", "sub4", "sub5", "sub6"]
        labels = [0, 1, 0, 1, 0, 1]
        
        folds = create_classification_stratified_folds(subjects, labels, k=3)
        self.assertEqual(len(folds), 3)
        
        # Each fold should have exactly 2 validation subjects (balanced)
        for i, (train_subjects, val_subjects) in enumerate(folds):
            self.assertEqual(len(val_subjects), 2)
            self.assertEqual(len(train_subjects), 4)
            
        # Test error case: too many folds for available data
        with self.assertRaises(ValueError):
            create_classification_stratified_folds(subjects, labels, k=7)
    
    def test_classification_folds_error_cases(self):
        """Test error handling for invalid inputs."""
        subjects = ["sub1", "sub2"]
        labels = [0, 1]
        
        # Test invalid k
        with self.assertRaises(ValueError):
            create_classification_stratified_folds(subjects, labels, k=1)
        
        # Test mismatched lengths
        with self.assertRaises(ValueError):
            create_classification_stratified_folds(subjects, [0], k=2)
        
        # Test insufficient subjects
        with self.assertRaises(ValueError):
            create_classification_stratified_folds(["sub1"], [0], k=2)


if __name__ == '__main__':
    unittest.main()
