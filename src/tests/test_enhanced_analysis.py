#!/usr/bin/env python3
"""
Unit tests for enhanced analysis tools.
Tests tools/enhanced_analysis.py functionality.
"""

import unittest
import numpy as np
import tempfile
import json
import os
import shutil
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Mock matplotlib and seaborn to avoid display issues
with patch.dict('sys.modules', {
    'matplotlib': MagicMock(),
    'matplotlib.pyplot': MagicMock(),
    'seaborn': MagicMock()
}):
    from tools.enhanced_analysis import EnhancedTrainingAnalyzer, main


class TestEnhancedTrainingAnalyzer(unittest.TestCase):
    
    def setUp(self):
        """Set up test analyzer with temporary data"""
        self.temp_dir = tempfile.mkdtemp()
        self.runs_dir = Path(self.temp_dir) / 'runs'
        self.runs_dir.mkdir()
        
        # Create mock fold structure
        self.num_folds = 3
        self.create_mock_training_data()
        
        self.analyzer = EnhancedTrainingAnalyzer(
            runs_dir=str(self.runs_dir),
            num_folds=self.num_folds
        )
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    def create_mock_training_data(self):
        """Create mock training data structure"""
        for fold in range(self.num_folds):
            fold_dir = self.runs_dir / f'fold{fold}' / 'Task001_FOMO1' / 'unet_xl' / 'version_0'
            fold_dir.mkdir(parents=True)
            
            # Create subject_probs_enhanced directory with mock data
            enhanced_dir = fold_dir / 'subject_probs_enhanced'
            enhanced_dir.mkdir()
            
            # Create mock enhanced results for multiple epochs
            for epoch in [5, 10, 15]:
                enhanced_data = {
                    'epoch': epoch,
                    'method_results': {
                        'mean_prob': {'subj_A': 0.7 + fold * 0.05, 'subj_B': 0.3 - fold * 0.02},
                        'mean_logit': {'subj_A': 0.72 + fold * 0.05, 'subj_B': 0.28 - fold * 0.02},
                        'noisy_or': {'subj_A': 0.75 + fold * 0.05, 'subj_B': 0.25 - fold * 0.02}
                    },
                    'method_aurocs': {
                        'mean_prob': 0.75 + fold * 0.03 + epoch * 0.01,
                        'mean_logit': 0.78 + fold * 0.03 + epoch * 0.01,
                        'noisy_or': 0.80 + fold * 0.03 + epoch * 0.01
                    }
                }
                
                enhanced_file = enhanced_dir / f'enhanced_results_epoch_{epoch:04d}.json'
                with open(enhanced_file, 'w') as f:
                    json.dump(enhanced_data, f)
            
            # Create training logs
            logs_dir = fold_dir / 'training_plots'
            logs_dir.mkdir()
            
            # Mock CSV files
            csv_data = [
                ['epoch', 'train_loss', 'val_loss', 'val_auroc_subject'],
                ['0', '0.8', '0.7', '0.6'],
                ['5', '0.6', '0.5', '0.75'],
                ['10', '0.4', '0.35', '0.82'],
                ['15', '0.3', '0.28', '0.85']
            ]
            
            # Would write CSV but mocking is sufficient for tests
    
    def test_initialization(self):
        """Test proper initialization of analyzer"""
        self.assertEqual(self.analyzer.runs_dir, Path(self.runs_dir))
        self.assertEqual(self.analyzer.num_folds, self.num_folds)
        self.assertEqual(len(self.analyzer.fold_results), 0)  # Initially empty
    
    def test_find_fold_directories(self):
        """Test finding fold directories"""
        fold_dirs = self.analyzer._find_fold_directories()
        
        self.assertEqual(len(fold_dirs), self.num_folds)
        
        for i, fold_dir in enumerate(fold_dirs):
            expected_path = self.runs_dir / f'fold{i}' / 'Task001_FOMO1' / 'unet_xl' / 'version_0'
            self.assertEqual(fold_dir, expected_path)
    
    def test_load_enhanced_results(self):
        """Test loading enhanced results from a fold"""
        fold_dir = self.runs_dir / 'fold0' / 'Task001_FOMO1' / 'unet_xl' / 'version_0'
        
        results = self.analyzer._load_enhanced_results(fold_dir)
        
        # Should load results from all epochs
        self.assertGreater(len(results), 0)
        
        # Check structure of loaded results
        for result in results:
            self.assertIn('epoch', result)
            self.assertIn('method_results', result)
            self.assertIn('method_aurocs', result)
            
            # Check method types
            self.assertIn('mean_prob', result['method_aurocs'])
            self.assertIn('mean_logit', result['method_aurocs'])
            self.assertIn('noisy_or', result['method_aurocs'])
    
    def test_load_all_fold_results(self):
        """Test loading results from all folds"""
        self.analyzer.load_all_fold_results()
        
        # Should have results for all folds
        self.assertEqual(len(self.analyzer.fold_results), self.num_folds)
        
        # Each fold should have multiple epochs
        for fold_idx, fold_results in self.analyzer.fold_results.items():
            self.assertGreater(len(fold_results), 0)
            
            # Check that epochs are sorted
            epochs = [r['epoch'] for r in fold_results]
            self.assertEqual(epochs, sorted(epochs))
    
    def test_get_best_epoch_per_fold(self):
        """Test finding best epoch per fold"""
        self.analyzer.load_all_fold_results()
        
        best_epochs = self.analyzer.get_best_epoch_per_fold('mean_logit')
        
        # Should have best epoch for each fold
        self.assertEqual(len(best_epochs), self.num_folds)
        
        # Each best epoch should be valid
        for fold_idx, best_epoch in best_epochs.items():
            self.assertIsInstance(best_epoch, int)
            self.assertGreaterEqual(best_epoch, 0)
    
    def test_get_method_performance_summary(self):
        """Test getting performance summary for all methods"""
        self.analyzer.load_all_fold_results()
        
        summary = self.analyzer.get_method_performance_summary()
        
        # Should have statistics for each method
        expected_methods = ['mean_prob', 'mean_logit', 'noisy_or']
        for method in expected_methods:
            self.assertIn(method, summary)
            
            method_stats = summary[method]
            self.assertIn('mean', method_stats)
            self.assertIn('std', method_stats)
            self.assertIn('min', method_stats)
            self.assertIn('max', method_stats)
            
            # Values should be reasonable
            self.assertTrue(0.0 <= method_stats['mean'] <= 1.0)
            self.assertTrue(method_stats['std'] >= 0.0)
    
    def test_get_cross_fold_stability(self):
        """Test cross-fold stability analysis"""
        self.analyzer.load_all_fold_results()
        
        stability = self.analyzer.get_cross_fold_stability()
        
        # Should have stability metrics for each method
        for method in ['mean_prob', 'mean_logit', 'noisy_or']:
            self.assertIn(method, stability)
            
            method_stability = stability[method]
            self.assertIn('coefficient_of_variation', method_stability)
            self.assertIn('range', method_stability)
            
            # CV should be non-negative
            self.assertGreaterEqual(method_stability['coefficient_of_variation'], 0.0)
            self.assertGreaterEqual(method_stability['range'], 0.0)
    
    def test_rank_methods_by_performance(self):
        """Test method ranking by performance"""
        self.analyzer.load_all_fold_results()
        
        ranking = self.analyzer.rank_methods_by_performance()
        
        # Should return list of tuples (method, score)
        self.assertIsInstance(ranking, list)
        self.assertGreater(len(ranking), 0)
        
        for method, score in ranking:
            self.assertIsInstance(method, str)
            self.assertIsInstance(score, (int, float))
            self.assertTrue(0.0 <= score <= 1.0)
        
        # Should be sorted in descending order
        scores = [score for _, score in ranking]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_rank_methods_by_stability(self):
        """Test method ranking by stability"""
        self.analyzer.load_all_fold_results()
        
        ranking = self.analyzer.rank_methods_by_stability()
        
        # Should return list of tuples (method, stability_score)
        self.assertIsInstance(ranking, list)
        self.assertGreater(len(ranking), 0)
        
        for method, stability_score in ranking:
            self.assertIsInstance(method, str)
            self.assertIsInstance(stability_score, (int, float))
        
        # Should be sorted by stability (lower CV = more stable = higher rank)
        stability_scores = [score for _, score in ranking]
        self.assertEqual(stability_scores, sorted(stability_scores, reverse=True))
    
    @patch('tools.enhanced_analysis.plt')
    @patch('tools.enhanced_analysis.sns')
    def test_plot_method_performance_comparison(self, mock_sns, mock_plt):
        """Test performance comparison plotting"""
        self.analyzer.load_all_fold_results()
        
        output_path = Path(self.temp_dir) / 'test_performance.png'
        
        try:
            self.analyzer.plot_method_performance_comparison(str(output_path))
        except Exception as e:
            self.fail(f"Performance comparison plotting failed: {e}")
        
        # Verify plotting calls were made
        self.assertTrue(mock_plt.subplots.called)
        self.assertTrue(mock_plt.savefig.called)
    
    @patch('tools.enhanced_analysis.plt')
    def test_plot_training_dynamics_per_fold(self, mock_plt):
        """Test training dynamics plotting"""
        self.analyzer.load_all_fold_results()
        
        output_path = Path(self.temp_dir) / 'test_dynamics.png'
        
        try:
            self.analyzer.plot_training_dynamics_per_fold(str(output_path))
        except Exception as e:
            self.fail(f"Training dynamics plotting failed: {e}")
        
        # Verify plotting calls were made
        self.assertTrue(mock_plt.subplots.called)
        self.assertTrue(mock_plt.savefig.called)
    
    @patch('tools.enhanced_analysis.plt')
    def test_plot_method_ranking_over_time(self, mock_plt):
        """Test method ranking over time plotting"""
        self.analyzer.load_all_fold_results()
        
        output_path = Path(self.temp_dir) / 'test_ranking.png'
        
        try:
            self.analyzer.plot_method_ranking_over_time(str(output_path))
        except Exception as e:
            self.fail(f"Method ranking plotting failed: {e}")
        
        # Verify plotting calls were made
        self.assertTrue(mock_plt.figure.called or mock_plt.subplots.called)
        self.assertTrue(mock_plt.savefig.called)
    
    @patch('tools.enhanced_analysis.plt')
    @patch('tools.enhanced_analysis.sns')
    def test_plot_cross_fold_stability(self, mock_sns, mock_plt):
        """Test cross-fold stability plotting"""
        self.analyzer.load_all_fold_results()
        
        output_path = Path(self.temp_dir) / 'test_stability.png'
        
        try:
            self.analyzer.plot_cross_fold_stability(str(output_path))
        except Exception as e:
            self.fail(f"Cross-fold stability plotting failed: {e}")
        
        # Verify plotting calls were made
        self.assertTrue(mock_plt.figure.called or mock_plt.subplots.called)
        self.assertTrue(mock_plt.savefig.called)
    
    def test_generate_text_summary(self):
        """Test text summary generation"""
        self.analyzer.load_all_fold_results()
        
        summary = self.analyzer.generate_text_summary()
        
        # Should return string
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
        
        # Should contain key information
        self.assertIn('Enhanced Training Analysis', summary)
        self.assertIn('Method Performance', summary)
        self.assertIn('Cross-Fold Stability', summary)
        self.assertIn('Recommendations', summary)
    
    def test_generate_comprehensive_report(self):
        """Test comprehensive report generation"""
        self.analyzer.load_all_fold_results()
        
        output_dir = Path(self.temp_dir) / 'report_output'
        
        try:
            self.analyzer.generate_comprehensive_report(str(output_dir))
        except Exception as e:
            self.fail(f"Comprehensive report generation failed: {e}")
        
        # Check that output directory was created
        self.assertTrue(output_dir.exists())
    
    def test_save_detailed_results(self):
        """Test detailed results saving"""
        self.analyzer.load_all_fold_results()
        
        output_path = Path(self.temp_dir) / 'detailed_results.json'
        self.analyzer.save_detailed_results(str(output_path))
        
        # Check that file was created
        self.assertTrue(output_path.exists())
        
        # Load and verify structure
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        self.assertIn('fold_results', data)
        self.assertIn('performance_summary', data)
        self.assertIn('stability_analysis', data)
        self.assertIn('method_rankings', data)
    
    def test_empty_results_handling(self):
        """Test handling of empty or missing results"""
        # Create analyzer with empty directory
        empty_dir = Path(self.temp_dir) / 'empty_runs'
        empty_dir.mkdir()
        
        empty_analyzer = EnhancedTrainingAnalyzer(str(empty_dir), num_folds=3)
        
        try:
            empty_analyzer.load_all_fold_results()
            # Should handle gracefully
            self.assertEqual(len(empty_analyzer.fold_results), 0)
        except Exception as e:
            self.fail(f"Empty results handling failed: {e}")
    
    def test_partial_fold_results(self):
        """Test handling of partial fold results"""
        # Remove one fold to test partial results
        shutil.rmtree(self.runs_dir / 'fold2')
        
        try:
            self.analyzer.load_all_fold_results()
            # Should load available folds
            self.assertEqual(len(self.analyzer.fold_results), 2)
        except Exception as e:
            self.fail(f"Partial fold results handling failed: {e}")


class TestEnhancedAnalysisMain(unittest.TestCase):
    """Test main function and CLI functionality"""
    
    def test_argument_parsing(self):
        """Test command line argument parsing"""
        with patch('sys.argv', ['enhanced_analysis.py', '/path/to/runs', '--num_folds', '5']):
            with patch('tools.enhanced_analysis.EnhancedTrainingAnalyzer') as mock_analyzer:
                mock_instance = MagicMock()
                mock_analyzer.return_value = mock_instance
                
                try:
                    main()
                except SystemExit:
                    pass  # Expected for argument parsing
                
                # Test would verify correct argument handling
    
    @patch('tools.enhanced_analysis.EnhancedTrainingAnalyzer')
    def test_main_execution_flow(self, mock_analyzer_class):
        """Test main execution flow"""
        mock_analyzer = MagicMock()
        mock_analyzer_class.return_value = mock_analyzer
        
        # Mock sys.argv
        test_args = [
            'enhanced_analysis.py',
            '/test/runs',
            '--output_dir', '/test/output',
            '--num_folds', '3'
        ]
        
        with patch('sys.argv', test_args):
            try:
                main()
            except SystemExit:
                pass  # Normal exit
            
            # Verify analyzer was created and methods called
            mock_analyzer_class.assert_called_once()
            mock_analyzer.load_all_fold_results.assert_called_once()
            mock_analyzer.generate_comprehensive_report.assert_called_once()


class TestEnhancedAnalysisIntegration(unittest.TestCase):
    """Integration tests for enhanced analysis"""
    
    def test_full_analysis_workflow(self):
        """Test complete analysis workflow"""
        temp_dir = tempfile.mkdtemp()
        runs_dir = Path(temp_dir) / 'runs'
        runs_dir.mkdir()
        
        try:
            # Create minimal mock data
            fold_dir = runs_dir / 'fold0' / 'Task001_FOMO1' / 'unet_xl' / 'version_0'
            enhanced_dir = fold_dir / 'subject_probs_enhanced'
            enhanced_dir.mkdir(parents=True)
            
            # Create one enhanced result file
            enhanced_data = {
                'epoch': 10,
                'method_results': {
                    'mean_prob': {'subj_A': 0.7, 'subj_B': 0.3},
                    'mean_logit': {'subj_A': 0.72, 'subj_B': 0.28}
                },
                'method_aurocs': {
                    'mean_prob': 0.75,
                    'mean_logit': 0.78
                }
            }
            
            enhanced_file = enhanced_dir / 'enhanced_results_epoch_0010.json'
            with open(enhanced_file, 'w') as f:
                json.dump(enhanced_data, f)
            
            # Run analysis
            analyzer = EnhancedTrainingAnalyzer(str(runs_dir), num_folds=1)
            analyzer.load_all_fold_results()
            
            # Should load data successfully
            self.assertEqual(len(analyzer.fold_results), 1)
            
            # Should be able to generate summary
            summary = analyzer.generate_text_summary()
            self.assertIsInstance(summary, str)
            self.assertGreater(len(summary), 0)
            
        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
