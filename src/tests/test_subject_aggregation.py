#!/usr/bin/env python3
"""
Unit tests for enhanced subject aggregation methods.
Tests all aggregation methods in src/utils/subject_aggregation.py
"""

import unittest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.subject_aggregation import SubjectAggregationSuite


class TestSubjectAggregation(unittest.TestCase):
    
    def setUp(self):
        """Set up test data with known patterns"""
        # Test data: 3 subjects, varying number of crops per subject
        self.subjects = ['subj_A', 'subj_B', 'subj_C']
        self.targets = {'subj_A': 1, 'subj_B': 0, 'subj_C': 1}
        
        # Subject A (positive): high confidence crops
        self.probs_A = torch.tensor([0.9, 0.85, 0.8])
        
        # Subject B (negative): low confidence crops  
        self.probs_B = torch.tensor([0.1, 0.15, 0.2, 0.05])
        
        # Subject C (positive): mixed confidence (sparse positive evidence)
        self.probs_C = torch.tensor([0.95, 0.1, 0.05, 0.15, 0.9])  # 2 strong positives, 3 negatives
        
        self.all_probs = {
            'subj_A': self.probs_A,
            'subj_B': self.probs_B, 
            'subj_C': self.probs_C
        }
        
        self.suite = SubjectAggregationSuite()
    
    def test_mean_prob_basic(self):
        """Test basic mean probability aggregation"""
        # Test individual subjects
        result_A = self.suite.mean_prob(self.probs_A)
        expected_A = torch.mean(self.probs_A)
        self.assertAlmostEqual(result_A.item(), expected_A.item(), places=5)
        
        result_B = self.suite.mean_prob(self.probs_B)
        expected_B = torch.mean(self.probs_B)
        self.assertAlmostEqual(result_B.item(), expected_B.item(), places=5)
    
    def test_mean_logit_vs_mean_prob(self):
        """Test that mean_logit differs from mean_prob and handles edge cases"""
        # For concentrated high probabilities, mean_logit should be higher
        high_probs = torch.tensor([0.9, 0.95, 0.92])
        mean_prob_result = self.suite.mean_prob(high_probs)
        mean_logit_result = self.suite.mean_logit(high_probs)
        
        # mean_logit should be higher for concentrated high probabilities
        self.assertGreater(mean_logit_result.item(), mean_prob_result.item())
        
        # Test edge case handling (should not produce NaN or inf)
        edge_probs = torch.tensor([0.999, 0.001, 0.5])
        result = self.suite.mean_logit(edge_probs)
        self.assertFalse(torch.isnan(result))
        self.assertFalse(torch.isinf(result))
    
    def test_noisy_or_sparse_evidence(self):
        """Test noisy_or aggregation for sparse positive evidence"""
        # Subject C has sparse positive evidence - noisy_or should capture this
        result_C = self.suite.noisy_or(self.probs_C)
        mean_result_C = self.suite.mean_prob(self.probs_C)
        
        # noisy_or should give higher probability than mean for sparse positive evidence
        self.assertGreater(result_C.item(), mean_result_C.item())
        
        # Test mathematical correctness: P(positive) = 1 - ∏(1 - p_i)
        expected = 1.0 - torch.prod(1.0 - self.probs_C)
        self.assertAlmostEqual(result_C.item(), expected.item(), places=5)
    
    def test_top_k_aggregation(self):
        """Test top-k aggregation methods"""
        probs = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3])  # sorted: [0.9, 0.8, 0.3, 0.2, 0.1]
        
        # Test top_k_3
        result_top3 = self.suite.top_k_3(probs)
        # Should average top 3 logits: logit(0.9), logit(0.8), logit(0.3)
        top_3_probs = torch.tensor([0.9, 0.8, 0.3])
        expected_top3 = self.suite.mean_logit(top_3_probs)
        self.assertAlmostEqual(result_top3.item(), expected_top3.item(), places=5)
        
        # Test top_k_5 (should be same as mean_logit since we have exactly 5 crops)
        result_top5 = self.suite.top_k_5(probs)
        expected_top5 = self.suite.mean_logit(probs)
        self.assertAlmostEqual(result_top5.item(), expected_top5.item(), places=5)
        
        # Test with fewer than k crops
        few_probs = torch.tensor([0.7, 0.8])
        result_few = self.suite.top_k_3(few_probs)
        expected_few = self.suite.mean_logit(few_probs)  # Should fallback to all crops
        self.assertAlmostEqual(result_few.item(), expected_few.item(), places=5)
    
    def test_weighted_entropy_aggregation(self):
        """Test entropy-weighted aggregation"""
        # High confidence crops should get more weight
        high_conf = torch.tensor([0.95, 0.9])  # Low entropy (high confidence)
        low_conf = torch.tensor([0.6, 0.4])   # High entropy (low confidence)
        
        result_high = self.suite.weighted_entropy(high_conf)
        result_low = self.suite.weighted_entropy(low_conf)
        
        # Should not produce NaN
        self.assertFalse(torch.isnan(result_high))
        self.assertFalse(torch.isnan(result_low))
        
        # Test mixed confidence - result should be reasonable
        mixed = torch.tensor([0.95, 0.5, 0.9])
        result_mixed = self.suite.weighted_entropy(mixed)
        self.assertTrue(0.0 <= result_mixed.item() <= 1.0)
    
    def test_robust_mean_outlier_removal(self):
        """Test robust mean removes outliers correctly"""
        # Data with outliers
        probs_with_outliers = torch.tensor([0.5, 0.6, 0.55, 0.05, 0.95])  # 0.05 and 0.95 are outliers
        
        robust_result = self.suite.robust_mean(probs_with_outliers)
        regular_result = self.suite.mean_prob(probs_with_outliers)
        
        # Robust mean should be different from regular mean
        self.assertNotAlmostEqual(robust_result.item(), regular_result.item(), places=3)
        
        # Result should be reasonable
        self.assertTrue(0.0 <= robust_result.item() <= 1.0)
    
    def test_consensus_aggregation(self):
        """Test consensus aggregation (fraction above threshold)"""
        # Test with known threshold crossing
        probs = torch.tensor([0.3, 0.7, 0.8, 0.2, 0.9])  # 3 out of 5 above 0.5
        result = self.suite.consensus(probs)
        expected = 3.0 / 5.0  # 60% consensus
        self.assertAlmostEqual(result.item(), expected, places=5)
        
        # Test edge cases
        all_low = torch.tensor([0.1, 0.2, 0.3])
        result_low = self.suite.consensus(all_low)
        self.assertEqual(result_low.item(), 0.0)
        
        all_high = torch.tensor([0.7, 0.8, 0.9])
        result_high = self.suite.consensus(all_high)
        self.assertEqual(result_high.item(), 1.0)
    
    def test_geometric_mean_aggregation(self):
        """Test geometric mean aggregation"""
        probs = torch.tensor([0.8, 0.6, 0.9])
        result = self.suite.geometric_mean(probs)
        
        # Geometric mean should be less than arithmetic mean for these values
        arithmetic_mean = torch.mean(probs)
        self.assertLess(result.item(), arithmetic_mean.item())
        
        # Mathematical correctness
        expected = torch.pow(torch.prod(probs), 1.0 / len(probs))
        self.assertAlmostEqual(result.item(), expected.item(), places=5)
    
    def test_max_prob_aggregation(self):
        """Test maximum probability aggregation"""
        probs = torch.tensor([0.3, 0.9, 0.5, 0.7])
        result = self.suite.max_prob(probs)
        expected = torch.max(probs)
        self.assertEqual(result.item(), expected.item())
    
    def test_aggregate_all_methods(self):
        """Test that all methods can be applied to all subjects"""
        all_results = self.suite.aggregate_all_methods(self.all_probs)
        
        # Check that all methods are present
        expected_methods = [
            'mean_prob', 'mean_logit', 'noisy_or', 'top_k_3', 'top_k_5',
            'weighted_entropy', 'robust_mean', 'consensus', 'geometric_mean', 'max_prob'
        ]
        
        for method in expected_methods:
            self.assertIn(method, all_results)
            
            # Check that all subjects have results for this method
            for subject in self.subjects:
                self.assertIn(subject, all_results[method])
                result = all_results[method][subject]
                
                # Results should be valid probabilities
                self.assertTrue(0.0 <= result <= 1.0)
                self.assertFalse(np.isnan(result))
                self.assertFalse(np.isinf(result))
    
    def test_compute_method_aurocs(self):
        """Test AUROC computation for all methods"""
        all_results = self.suite.aggregate_all_methods(self.all_probs)
        aurocs = self.suite.compute_method_aurocs(all_results, self.targets)
        
        # Check that AUROCs are computed for all methods
        for method in all_results.keys():
            self.assertIn(method, aurocs)
            auroc = aurocs[method]
            
            # AUROC should be between 0 and 1
            self.assertTrue(0.0 <= auroc <= 1.0)
            self.assertFalse(np.isnan(auroc))
    
    def test_empty_crops_handling(self):
        """Test handling of edge cases like empty crop lists"""
        # This should not happen in practice, but test robustness
        empty_probs = torch.tensor([])
        
        # Most methods should handle this gracefully or raise appropriate errors
        with self.assertRaises((RuntimeError, ValueError)):
            self.suite.mean_prob(empty_probs)
    
    def test_single_crop_handling(self):
        """Test handling of subjects with single crops"""
        single_crop = torch.tensor([0.7])
        
        # All methods should work with single crops
        methods_to_test = [
            'mean_prob', 'mean_logit', 'noisy_or', 'max_prob',
            'geometric_mean', 'consensus'
        ]
        
        for method_name in methods_to_test:
            method = getattr(self.suite, method_name)
            result = method(single_crop)
            self.assertTrue(0.0 <= result.item() <= 1.0)
    
    def test_method_consistency(self):
        """Test that methods produce consistent results across calls"""
        # Same input should always produce same output
        probs = torch.tensor([0.3, 0.7, 0.5])
        
        for method_name in ['mean_prob', 'mean_logit', 'noisy_or']:
            method = getattr(self.suite, method_name)
            result1 = method(probs)
            result2 = method(probs)
            self.assertEqual(result1.item(), result2.item())
    
    def test_aggregation_ordering(self):
        """Test that different methods rank subjects consistently with expected patterns"""
        all_results = self.suite.aggregate_all_methods(self.all_probs)
        
        # For our test data:
        # Subject A: consistently high probs -> should rank high
        # Subject B: consistently low probs -> should rank low  
        # Subject C: sparse high evidence -> noisy_or should rank higher than mean_prob
        
        # Test noisy_or vs mean_prob for Subject C (sparse evidence)
        noisy_or_C = all_results['noisy_or']['subj_C']
        mean_prob_C = all_results['mean_prob']['subj_C']
        self.assertGreater(noisy_or_C, mean_prob_C, 
                          "noisy_or should rank higher than mean_prob for sparse positive evidence")
        
        # Test that Subject A ranks higher than Subject B for most methods
        for method in ['mean_prob', 'mean_logit', 'max_prob']:
            result_A = all_results[method]['subj_A']
            result_B = all_results[method]['subj_B']
            self.assertGreater(result_A, result_B, 
                              f"{method} should rank positive subject A higher than negative subject B")


if __name__ == '__main__':
    unittest.main()
