"""
Enhanced subject-level aggregation methods for multi-crop medical imaging predictions.
Implements multiple aggregation strategies and automatic method selection.
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import json
import warnings

class SubjectAggregationSuite:
    """
    Comprehensive suite of subject-level aggregation methods for converting
    multiple crop predictions into single subject-level predictions.
    """
    
    def __init__(self, methods: Optional[List[str]] = None):
        """
        Initialize aggregation suite.
        
        Args:
            methods: List of aggregation methods to use. If None, uses all available.
        """
        self.available_methods = {
            'mean_prob': self.mean_probability,
            'mean_logit': self.mean_logit,
            'noisy_or': self.noisy_or,
            'top_k_3': lambda probs: self.top_k_logit(probs, k=3),
            'top_k_5': lambda probs: self.top_k_logit(probs, k=5),
            'max_prob': lambda probs: max(probs) if probs else 0.0,
            'weighted_entropy': self.entropy_weighted_mean,
            'robust_mean': self.robust_mean,
            'consensus': self.consensus_voting,
            'quantile_75': lambda probs: np.percentile(probs, 75) if probs else 0.0,
            'geometric_mean': self.geometric_mean
        }
        
        if methods is None:
            self.methods = list(self.available_methods.keys())
        else:
            self.methods = [m for m in methods if m in self.available_methods]
            
        self.method_performance_history = defaultdict(list)
        
    def aggregate_all_methods(self, subject_crop_probs: Dict[str, List[float]], 
                             subject_targets: Dict[str, int]) -> Dict[str, Dict[str, float]]:
        """
        Apply all aggregation methods to subject crop probabilities.
        
        Args:
            subject_crop_probs: Dict mapping subject_id -> list of crop probabilities
            subject_targets: Dict mapping subject_id -> target label
            
        Returns:
            Dict with structure: {subject_id: {method_name: aggregated_prob, 'target': label}}
        """
        results = {}
        
        for subject_id, crop_probs in subject_crop_probs.items():
            if not crop_probs:
                continue
                
            subject_result = {'target': subject_targets.get(subject_id, 0)}
            
            for method_name in self.methods:
                try:
                    aggregated_prob = self.available_methods[method_name](crop_probs)
                    subject_result[method_name] = float(aggregated_prob)
                except Exception as e:
                    warnings.warn(f"Error in {method_name} for subject {subject_id}: {e}")
                    subject_result[method_name] = 0.0
            
            results[subject_id] = subject_result
            
        return results
    
    def mean_probability(self, probs: List[float]) -> float:
        """Standard mean of probabilities (current method)."""
        return np.mean(probs) if probs else 0.0
    
    def mean_logit(self, probs: List[float]) -> float:
        """
        Mean of logits aggregation - often better than mean of probabilities
        for handling concentrated evidence or conflicting predictions.
        """
        if not probs:
            return 0.0
            
        # Convert probabilities to logits
        logits = []
        for p in probs:
            # Clamp probability to avoid log(0) or log(1-0)
            p_clamped = np.clip(p, 1e-8, 1 - 1e-8)
            logit = np.log(p_clamped / (1 - p_clamped))
            logits.append(logit)
        
        # Average logits and convert back to probability
        mean_logit_val = np.mean(logits)
        return 1.0 / (1.0 + np.exp(-mean_logit_val))
    
    def noisy_or(self, probs: List[float]) -> float:
        """
        Noisy-OR aggregation: assumes positive evidence from any crop
        is indicative of positive subject. Good for sparse positive findings.
        P(positive) = 1 - ∏(1 - p_i)
        """
        if not probs:
            return 0.0
        
        neg_prob_product = 1.0
        for p in probs:
            neg_prob_product *= (1.0 - p)
        
        return 1.0 - neg_prob_product
    
    def top_k_logit(self, probs: List[float], k: int = 3) -> float:
        """
        Top-K logit aggregation: focus on the strongest evidence
        by averaging the top-k crop logits.
        """
        if not probs:
            return 0.0
            
        if len(probs) <= k:
            return self.mean_logit(probs)
        
        # Get top-k probabilities
        top_k_probs = sorted(probs, reverse=True)[:k]
        return self.mean_logit(top_k_probs)
    
    def entropy_weighted_mean(self, probs: List[float]) -> float:
        """
        Entropy-weighted mean: weight crops by confidence (inverse entropy).
        More confident predictions get higher weight.
        """
        if not probs:
            return 0.0
        
        weights = []
        for p in probs:
            # Calculate entropy (uncertainty)
            p_clamped = np.clip(p, 1e-8, 1 - 1e-8)
            entropy = -(p_clamped * np.log(p_clamped) + (1 - p_clamped) * np.log(1 - p_clamped))
            # Convert to confidence weight (lower entropy = higher confidence = higher weight)
            confidence = 1.0 - entropy / np.log(2)  # Normalize by max entropy
            weights.append(confidence)
        
        # Weighted average
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize weights
        
        return np.sum(weights * np.array(probs))
    
    def robust_mean(self, probs: List[float], trim_fraction: float = 0.1) -> float:
        """
        Robust mean: trim extreme values before averaging to reduce
        impact of outlier predictions.
        """
        if not probs:
            return 0.0
        
        if len(probs) <= 2:
            return np.mean(probs)
        
        # Sort and trim extremes
        sorted_probs = sorted(probs)
        n_trim = max(1, int(len(probs) * trim_fraction))
        
        trimmed_probs = sorted_probs[n_trim:-n_trim] if n_trim < len(probs) // 2 else sorted_probs
        
        return np.mean(trimmed_probs)
    
    def consensus_voting(self, probs: List[float], threshold: float = 0.5) -> float:
        """
        Consensus voting: fraction of crops that predict positive class.
        Returns the proportion of crops above threshold.
        """
        if not probs:
            return 0.0
        
        positive_votes = sum(1 for p in probs if p > threshold)
        return positive_votes / len(probs)
    
    def geometric_mean(self, probs: List[float]) -> float:
        """
        Geometric mean of probabilities.
        More conservative than arithmetic mean.
        """
        if not probs:
            return 0.0
        
        # Handle zero probabilities
        probs_clamped = [max(p, 1e-8) for p in probs]
        return np.prod(probs_clamped) ** (1.0 / len(probs_clamped))
    
    def compute_method_aurocs(self, results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Compute AUROC for each aggregation method.
        
        Args:
            results: Results from aggregate_all_methods
            
        Returns:
            Dict mapping method_name -> AUROC score
        """
        method_aurocs = {}
        
        # Extract targets
        subjects = list(results.keys())
        targets = [results[sid]['target'] for sid in subjects]
        
        # Check if we have both classes
        if len(set(targets)) < 2:
            # Return NaN for all methods if only one class present
            return {method: float('nan') for method in self.methods}
        
        # Compute AUROC for each method
        for method in self.methods:
            try:
                probs = [results[sid][method] for sid in subjects]
                auroc = self._compute_auroc(probs, targets)
                method_aurocs[method] = auroc
            except Exception as e:
                warnings.warn(f"Error computing AUROC for {method}: {e}")
                method_aurocs[method] = float('nan')
        
        # Store in history
        for method, auroc in method_aurocs.items():
            if not np.isnan(auroc):
                self.method_performance_history[method].append(auroc)
        
        return method_aurocs
    
    def select_best_method(self, results: Dict[str, Dict[str, float]], 
                          strategy: str = 'current_best') -> str:
        """
        Select the best performing aggregation method.
        
        Args:
            results: Results from aggregate_all_methods
            strategy: Selection strategy ('current_best', 'historical_best', 'ensemble')
            
        Returns:
            Name of the best method
        """
        if strategy == 'current_best':
            method_aurocs = self.compute_method_aurocs(results)
            # Filter out NaN values
            valid_aurocs = {k: v for k, v in method_aurocs.items() if not np.isnan(v)}
            if not valid_aurocs:
                return 'mean_logit'  # Default fallback
            return max(valid_aurocs.keys(), key=lambda k: valid_aurocs[k])
        
        elif strategy == 'historical_best':
            # Use historical performance
            historical_means = {}
            for method in self.methods:
                if method in self.method_performance_history:
                    scores = self.method_performance_history[method]
                    if scores:
                        historical_means[method] = np.mean(scores)
            
            if not historical_means:
                return 'mean_logit'  # Default fallback
            return max(historical_means.keys(), key=lambda k: historical_means[k])
        
        elif strategy == 'ensemble':
            # Return multiple methods for ensemble
            return ['mean_logit', 'noisy_or', 'top_k_3']
        
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")
    
    def get_method_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all methods based on historical performance.
        
        Returns:
            Dict with method names and their performance statistics
        """
        summary = {}
        
        for method in self.methods:
            if method in self.method_performance_history:
                scores = self.method_performance_history[method]
                if scores:
                    summary[method] = {
                        'mean_auroc': np.mean(scores),
                        'std_auroc': np.std(scores),
                        'best_auroc': np.max(scores),
                        'n_evaluations': len(scores)
                    }
                else:
                    summary[method] = {
                        'mean_auroc': float('nan'),
                        'std_auroc': float('nan'),
                        'best_auroc': float('nan'),
                        'n_evaluations': 0
                    }
        
        return summary
    
    def _compute_auroc(self, probs: List[float], targets: List[int]) -> float:
        """Compute AUROC score."""
        try:
            # Try using sklearn if available
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(targets, probs)
        except ImportError:
            # Fallback: PyTorch implementation
            try:
                from torchmetrics.classification import AUROC
                auroc_metric = AUROC(task="binary")
                probs_tensor = torch.tensor(probs, dtype=torch.float32)
                targets_tensor = torch.tensor(targets, dtype=torch.int64)
                return float(auroc_metric(probs_tensor, targets_tensor).item())
            except ImportError:
                # Last resort: simple approximation
                warnings.warn("Neither sklearn nor torchmetrics available. Using approximation.")
                return self._simple_auroc_approximation(probs, targets)
    
    def _simple_auroc_approximation(self, probs: List[float], targets: List[int]) -> float:
        """Simple AUROC approximation when libraries aren't available."""
        # Very basic approximation - not recommended for production
        pos_probs = [probs[i] for i, t in enumerate(targets) if t == 1]
        neg_probs = [probs[i] for i, t in enumerate(targets) if t == 0]
        
        if not pos_probs or not neg_probs:
            return float('nan')
        
        # Count pairs where positive example has higher probability
        correct_pairs = 0
        total_pairs = 0
        
        for pos_prob in pos_probs:
            for neg_prob in neg_probs:
                total_pairs += 1
                if pos_prob > neg_prob:
                    correct_pairs += 1
                elif pos_prob == neg_prob:
                    correct_pairs += 0.5
        
        return correct_pairs / total_pairs if total_pairs > 0 else 0.5


class SubjectAggregationCallback:
    """
    Lightning callback for integrating enhanced subject aggregation 
    with the training process.
    """
    
    def __init__(self, aggregation_suite: SubjectAggregationSuite, 
                 save_results: bool = True):
        """
        Initialize callback.
        
        Args:
            aggregation_suite: The aggregation suite to use
            save_results: Whether to save detailed results to disk
        """
        self.aggregation_suite = aggregation_suite
        self.save_results = save_results
        
    def process_subject_predictions(self, subject_crop_data: Dict[str, List], 
                                  subject_targets: Dict[str, int],
                                  epoch: int, version_dir: str) -> Dict[str, float]:
        """
        Process subject predictions through all aggregation methods.
        
        Args:
            subject_crop_data: Dict mapping subject_id -> [sum_probs, count, target]
            subject_targets: Dict mapping subject_id -> target
            epoch: Current epoch
            version_dir: Directory to save results
            
        Returns:
            Dict of method_name -> AUROC for logging
        """
        # Convert from current format to crop probabilities
        subject_crop_probs = {}
        for subject_id, (sum_probs, count, target) in subject_crop_data.items():
            # We need to reconstruct individual probabilities
            # For now, approximate with uniform distribution around the mean
            mean_prob = sum_probs / count if count > 0 else 0.0
            
            # Create synthetic individual probabilities
            # This is a limitation of the current aggregation system
            # Ideally, we'd track individual probabilities
            crop_probs = [mean_prob] * int(count)  # Simplified
            subject_crop_probs[subject_id] = crop_probs
        
        # Apply all aggregation methods
        results = self.aggregation_suite.aggregate_all_methods(
            subject_crop_probs, subject_targets
        )
        
        # Compute AUROCs for each method
        method_aurocs = self.aggregation_suite.compute_method_aurocs(results)
        
        # Save detailed results if requested
        if self.save_results:
            self._save_detailed_results(results, method_aurocs, epoch, version_dir)
        
        # Store results for visualization
        # This will be picked up by the training visualizer
        return method_aurocs, results
    
    def _save_detailed_results(self, results: Dict, method_aurocs: Dict, 
                              epoch: int, version_dir: str):
        """Save detailed aggregation results to disk."""
        try:
            import os
            results_dir = os.path.join(version_dir, "aggregation_results")
            os.makedirs(results_dir, exist_ok=True)
            
            # Save comprehensive results
            save_data = {
                'epoch': epoch,
                'method_aurocs': method_aurocs,
                'subject_results': results,
                'method_summary': self.aggregation_suite.get_method_summary(),
                'best_method': self.aggregation_suite.select_best_method(results)
            }
            
            results_path = os.path.join(results_dir, f"aggregation_epoch_{epoch:04d}.json")
            with open(results_path, 'w') as f:
                json.dump(save_data, f, indent=2)
                
        except Exception as e:
            warnings.warn(f"Error saving aggregation results: {e}")


def create_enhanced_aggregation_system(methods: Optional[List[str]] = None,
                                     enable_visualization: bool = True) -> Tuple[SubjectAggregationSuite, SubjectAggregationCallback]:
    """
    Factory function to create a complete enhanced aggregation system.
    
    Args:
        methods: List of aggregation methods to use
        enable_visualization: Whether to enable detailed result saving
        
    Returns:
        Tuple of (aggregation_suite, callback)
    """
    suite = SubjectAggregationSuite(methods=methods)
    callback = SubjectAggregationCallback(suite, save_results=enable_visualization)
    
    return suite, callback
