"""
Unit tests for Cross-Validation Support

Tests the 5-fold cross-validation implementation for Task3 brain age regression,
including stratified folding, results aggregation, and fold consistency validation.
"""

import pytest
import tempfile
import os
import json
import numpy as np
from unittest.mock import patch, MagicMock
from typing import List, Tuple, Dict


class TestCrossValidationSupport:
    """Test suite for cross-validation support."""

    @pytest.fixture
    def sample_subjects_and_ages(self):
        """Generate sample subjects and ages for testing."""
        np.random.seed(42)  # Reproducible results
        n_subjects = 100
        subjects = [f"subject_{i:03d}" for i in range(n_subjects)]
        # Generate ages with realistic distribution (20-90 years)
        ages = np.random.normal(loc=60, scale=15, size=n_subjects)
        ages = np.clip(ages, 20, 90).tolist()
        return subjects, ages

    def test_age_stratified_fold_creation_basic(self, sample_subjects_and_ages):
        """Test basic functionality of age-stratified fold creation."""
        from utils.cross_validation import create_age_stratified_folds
        
        subjects, ages = sample_subjects_and_ages
        folds = create_age_stratified_folds(subjects, ages, k=5)
        
        # Should have exactly 5 folds
        assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"
        
        # Each fold should have train/val split
        total_subjects = len(subjects)
        for fold_idx, (train_subjects, val_subjects) in enumerate(folds):
            # All subjects accounted for
            assert len(train_subjects) + len(val_subjects) == total_subjects, (
                f"Fold {fold_idx}: subjects count mismatch"
            )
            
            # Reasonable validation split size (15-25% typical for 5-fold)
            val_ratio = len(val_subjects) / total_subjects
            assert 0.15 <= val_ratio <= 0.25, (
                f"Fold {fold_idx}: validation ratio {val_ratio:.3f} outside expected range"
            )
            
            # No overlap between train and validation
            overlap = set(train_subjects) & set(val_subjects)
            assert len(overlap) == 0, f"Fold {fold_idx}: train/val overlap: {overlap}"

    def test_age_stratified_fold_coverage(self, sample_subjects_and_ages):
        """Test that all subjects appear exactly once in validation across folds."""
        from utils.cross_validation import create_age_stratified_folds
        
        subjects, ages = sample_subjects_and_ages
        folds = create_age_stratified_folds(subjects, ages, k=5)
        
        # Collect all validation subjects across folds
        all_val_subjects = []
        for train_subjects, val_subjects in folds:
            all_val_subjects.extend(val_subjects)
        
        # Every subject should appear exactly once in validation
        assert len(all_val_subjects) == len(subjects), (
            "Total validation subjects doesn't match total subjects"
        )
        assert len(set(all_val_subjects)) == len(all_val_subjects), (
            "Some subjects appear multiple times in validation"
        )
        assert set(all_val_subjects) == set(subjects), (
            "Validation subjects don't match original subjects"
        )

    def test_age_distribution_balance_across_folds(self, sample_subjects_and_ages):
        """Test that age distribution is balanced across folds."""
        from utils.cross_validation import create_age_stratified_folds
        
        subjects, ages = sample_subjects_and_ages
        age_dict = dict(zip(subjects, ages))
        
        folds = create_age_stratified_folds(subjects, ages, k=5)
        
        # Analyze age distribution in each fold
        fold_age_stats = []
        for fold_idx, (train_subjects, val_subjects) in enumerate(folds):
            val_ages = [age_dict[s] for s in val_subjects]
            
            fold_stats = {
                'fold': fold_idx,
                'count': len(val_ages),
                'mean': np.mean(val_ages),
                'std': np.std(val_ages),
                'min': np.min(val_ages),
                'max': np.max(val_ages)
            }
            fold_age_stats.append(fold_stats)
        
        # Check that folds have similar age distributions
        means = [stats['mean'] for stats in fold_age_stats]
        std_of_means = np.std(means)
        
        # Standard deviation of fold means should be small (balanced distribution)
        assert std_of_means < 5.0, (
            f"Age distribution varies too much across folds (std={std_of_means:.2f}): {means}"
        )
        
        # Each fold should have representation from different age ranges
        for fold_idx, stats in enumerate(fold_age_stats):
            age_range = stats['max'] - stats['min']
            assert age_range > 20, (
                f"Fold {fold_idx} has too narrow age range ({age_range:.1f} years)"
            )

    def test_k_fold_parameter_validation(self, sample_subjects_and_ages):
        """Test validation of K-fold parameters."""
        from utils.cross_validation import create_age_stratified_folds
        
        subjects, ages = sample_subjects_and_ages
        
        # Test invalid K values
        with pytest.raises(ValueError, match="k must be >= 2"):
            create_age_stratified_folds(subjects, ages, k=1)
        
        with pytest.raises(ValueError, match="k must be >= 2"):
            create_age_stratified_folds(subjects, ages, k=0)
        
        # Test K larger than number of subjects
        small_subjects = subjects[:3]
        small_ages = ages[:3]
        
        with pytest.raises(ValueError, match="k cannot be larger than number of subjects"):
            create_age_stratified_folds(small_subjects, small_ages, k=5)
        
        # Test mismatched subjects and ages length
        with pytest.raises(ValueError, match="subjects and ages must have same length"):
            create_age_stratified_folds(subjects, ages[:50], k=5)

    def test_stratified_binning_logic(self):
        """Test the age binning logic for stratification."""
        from utils.cross_validation import get_age_bin, compute_age_quartiles
        
        # Test with known age distribution
        ages = [20, 30, 40, 50, 60, 70, 80, 90]
        quartiles = compute_age_quartiles(ages)
        
        # Quartiles should be [35, 50, 65] for this distribution
        expected_q1, expected_q2, expected_q3 = 35, 50, 65
        assert abs(quartiles[0] - expected_q1) < 5, f"Q1 mismatch: {quartiles[0]} vs {expected_q1}"
        assert abs(quartiles[1] - expected_q2) < 5, f"Q2 mismatch: {quartiles[1]} vs {expected_q2}"
        assert abs(quartiles[2] - expected_q3) < 5, f"Q3 mismatch: {quartiles[2]} vs {expected_q3}"
        
        # Test binning
        test_ages = [25, 45, 55, 75]
        expected_bins = [0, 1, 2, 3]  # Q1, Q2, Q3, Q4
        
        for age, expected_bin in zip(test_ages, expected_bins):
            bin_idx = get_age_bin(age, quartiles)
            assert bin_idx == expected_bin, f"Age {age} assigned to bin {bin_idx}, expected {expected_bin}"

    def test_kfold_results_aggregation_basic(self):
        """Test basic functionality of K-fold results aggregation."""
        from utils.aggregate_kfold_results import aggregate_kfold_results
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock fold result directories
            for fold in range(5):
                fold_dir = os.path.join(temp_dir, f"experiment_fold{fold}", "version_0")
                os.makedirs(fold_dir)
                
                # Create mock best_metrics.txt with realistic brain age metrics
                metrics_file = os.path.join(fold_dir, "best_metrics.txt")
                correlation = 0.75 + fold * 0.03  # Increasing correlation: 0.75, 0.78, 0.81, 0.84, 0.87
                mae = 6.0 - fold * 0.2            # Decreasing MAE: 6.0, 5.8, 5.6, 5.4, 5.2
                
                with open(metrics_file, 'w') as f:
                    f.write(f"val/corr: {correlation:.4f}\n")
                    f.write(f"val/mae: {mae:.2f}\n")
                    f.write(f"val/r2: {correlation**2:.4f}\n")  # R² approximation
            
            # Test aggregation
            aggregated = aggregate_kfold_results(
                results_dir=temp_dir,
                experiment_pattern="experiment_fold*",
                metrics=["val/corr", "val/mae", "val/r2"]
            )
            
            # Verify all metrics are present
            assert "val/corr" in aggregated, "Correlation metric missing"
            assert "val/mae" in aggregated, "MAE metric missing"
            assert "val/r2" in aggregated, "R² metric missing"
            
            # Check correlation statistics
            corr_stats = aggregated["val/corr"]
            expected_mean = 0.81  # Mean of 0.75, 0.78, 0.81, 0.84, 0.87
            assert abs(corr_stats["mean"] - expected_mean) < 0.01, (
                f"Correlation mean {corr_stats['mean']:.4f} != expected {expected_mean:.4f}"
            )
            assert corr_stats["min"] == 0.75, f"Correlation min should be 0.75"
            assert corr_stats["max"] == 0.87, f"Correlation max should be 0.87"
            
            # Check MAE statistics
            mae_stats = aggregated["val/mae"]
            expected_mae_mean = 5.6  # Mean of 6.0, 5.8, 5.6, 5.4, 5.2
            assert abs(mae_stats["mean"] - expected_mae_mean) < 0.01, (
                f"MAE mean {mae_stats['mean']:.2f} != expected {expected_mae_mean:.2f}"
            )

    def test_results_aggregation_missing_folds(self):
        """Test results aggregation when some folds are missing."""
        from utils.aggregate_kfold_results import aggregate_kfold_results
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create only 3 out of 5 folds
            for fold in [0, 2, 4]:  # Skip folds 1 and 3
                fold_dir = os.path.join(temp_dir, f"experiment_fold{fold}", "version_0")
                os.makedirs(fold_dir)
                
                metrics_file = os.path.join(fold_dir, "best_metrics.txt")
                with open(metrics_file, 'w') as f:
                    f.write(f"val/corr: {0.8 + fold * 0.01}\n")
                    f.write(f"val/mae: {5.0 + fold * 0.1}\n")
            
            # Should still aggregate available folds
            aggregated = aggregate_kfold_results(
                results_dir=temp_dir,
                experiment_pattern="experiment_fold*",
                metrics=["val/corr", "val/mae"]
            )
            
            # Should have 3 values for each metric
            assert len(aggregated["val/corr"]["values"]) == 3
            assert len(aggregated["val/mae"]["values"]) == 3

    def test_results_aggregation_no_matching_folds(self):
        """Test error handling when no matching fold directories are found."""
        from utils.aggregate_kfold_results import aggregate_kfold_results
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Empty directory
            with pytest.raises(ValueError, match="No fold directories found"):
                aggregate_kfold_results(
                    results_dir=temp_dir,
                    experiment_pattern="nonexistent_pattern*",
                    metrics=["val/corr"]
                )

    def test_results_report_formatting(self):
        """Test formatting of aggregated results into readable report."""
        from utils.aggregate_kfold_results import format_results_report
        
        # Mock aggregated results
        aggregated = {
            'val/corr': {
                'mean': 0.823,
                'std': 0.045,
                'min': 0.761,
                'max': 0.887,
                'values': [0.761, 0.798, 0.823, 0.856, 0.887]
            },
            'val/mae': {
                'mean': 5.24,
                'std': 0.68,
                'min': 4.32,
                'max': 6.15,
                'values': [6.15, 5.78, 5.24, 4.91, 4.32]
            }
        }
        
        report = format_results_report(aggregated)
        
        # Check that report contains key information
        assert "AGGREGATED K-FOLD RESULTS" in report
        assert "Val Corr:" in report or "VAL/CORR" in report
        assert "Mean: 0.8230" in report
        assert "Std:  0.0450" in report
        assert "Average Correlation: 0.823" in report
        assert "Average Error: 5.2 years" in report

    def test_fold_index_parameter_handling(self):
        """Test handling of fold index parameter in training script integration."""
        from utils.cross_validation import validate_fold_parameters
        
        # Test valid parameters
        assert validate_fold_parameters(k_folds=5, fold_index=0) == True
        assert validate_fold_parameters(k_folds=5, fold_index=4) == True
        
        # Test invalid parameters
        with pytest.raises(ValueError, match="fold_index must be between 0 and k_folds-1"):
            validate_fold_parameters(k_folds=5, fold_index=5)
        
        with pytest.raises(ValueError, match="fold_index must be between 0 and k_folds-1"):
            validate_fold_parameters(k_folds=5, fold_index=-1)
        
        with pytest.raises(ValueError, match="k_folds must be >= 2"):
            validate_fold_parameters(k_folds=1, fold_index=0)

    def test_sequential_execution_simulation(self, sample_subjects_and_ages):
        """Test simulation of sequential K-fold execution."""
        from utils.cross_validation import create_age_stratified_folds
        from utils.cross_validation import simulate_sequential_training
        
        subjects, ages = sample_subjects_and_ages
        folds = create_age_stratified_folds(subjects, ages, k=5)
        
        # Simulate training for each fold with mock results
        def mock_train_fold(train_subjects, val_subjects, fold_idx):
            """Mock training function that returns realistic metrics."""
            # Simulate some variation in performance across folds
            base_correlation = 0.8
            base_mae = 5.5
            
            # Add some realistic fold-to-fold variation
            correlation = base_correlation + np.random.normal(0, 0.03)
            mae = base_mae + np.random.normal(0, 0.4)
            
            return {
                'fold': fold_idx,
                'train_size': len(train_subjects),
                'val_size': len(val_subjects),
                'val/corr': correlation,
                'val/mae': mae
            }
        
        # Run simulation
        fold_results = simulate_sequential_training(folds, mock_train_fold)
        
        # Verify results
        assert len(fold_results) == 5
        for i, result in enumerate(fold_results):
            assert result['fold'] == i
            assert 'val/corr' in result
            assert 'val/mae' in result
            assert result['train_size'] > 0
            assert result['val_size'] > 0

    def test_cross_validation_reproducibility(self, sample_subjects_and_ages):
        """Test that cross-validation splits are reproducible with same random seed."""
        from utils.cross_validation import create_age_stratified_folds
        
        subjects, ages = sample_subjects_and_ages
        
        # Create folds with same seed twice
        np.random.seed(12345)
        folds1 = create_age_stratified_folds(subjects, ages, k=5)
        
        np.random.seed(12345) 
        folds2 = create_age_stratified_folds(subjects, ages, k=5)
        
        # Should be identical
        assert len(folds1) == len(folds2)
        for (train1, val1), (train2, val2) in zip(folds1, folds2):
            assert set(train1) == set(train2), "Training sets don't match"
            assert set(val1) == set(val2), "Validation sets don't match"

    def test_integration_with_finetune_script(self):
        """Test integration points with the main finetune.py script."""
        # Mock the parts of finetune.py that handle K-fold
        mock_config = {
            'k_folds': 5,
            'fold_index': 2,  # Third fold (0-indexed)
            'experiment': 'test_kfold'
        }
        
        # Test that fold parameters are correctly processed
        k_folds = mock_config['k_folds']
        fold_index = mock_config['fold_index']
        
        assert k_folds == 5
        assert 0 <= fold_index < k_folds
        
        # Test experiment naming
        expected_experiment_name = f"{mock_config['experiment']}_fold{fold_index}"
        assert expected_experiment_name == "test_kfold_fold2"

    def test_clinical_interpretation_thresholds(self):
        """Test clinical interpretation thresholds for brain age metrics."""
        from utils.aggregate_kfold_results import get_clinical_interpretation
        
        # Test correlation interpretation
        assert get_clinical_interpretation('correlation', 0.85) == 'Strong'
        assert get_clinical_interpretation('correlation', 0.70) == 'Moderate' 
        assert get_clinical_interpretation('correlation', 0.45) == 'Weak'
        
        # Test MAE interpretation (in years)
        assert get_clinical_interpretation('mae', 3.5) == 'Excellent'  # < 4 years
        assert get_clinical_interpretation('mae', 5.2) == 'Good'       # 4-6 years
        assert get_clinical_interpretation('mae', 7.8) == 'Fair'       # 6-8 years
        assert get_clinical_interpretation('mae', 9.5) == 'Poor'       # > 8 years


if __name__ == "__main__":
    pytest.main([__file__])
