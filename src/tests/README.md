# Enhanced Features Test Suite

This directory contains comprehensive unit tests for all enhanced training features implemented in the FOMO25 baseline codebase.

## Test Coverage

### 🎯 Subject Aggregation (`test_subject_aggregation.py`)
Tests for multiple subject-level aggregation methods:
- **Basic aggregation methods**: mean_prob, mean_logit, noisy_or
- **Advanced methods**: top_k, weighted_entropy, robust_mean, consensus, geometric_mean, max_prob
- **Edge cases**: empty crops, single crops, outlier handling
- **Mathematical correctness**: verifies formulas and expected behaviors
- **Performance ranking**: tests that methods rank subjects appropriately

**Key Test Cases:**
- `test_mean_logit_vs_mean_prob`: Verifies mean_logit performs better for concentrated evidence
- `test_noisy_or_sparse_evidence`: Tests noisy_or superiority for sparse positive findings
- `test_aggregation_ordering`: Validates method ranking consistency

### 🛑 Robust Early Stopping (`test_robust_early_stopping.py`)
Tests for enhanced early stopping mechanisms:
- **Ensemble metrics**: combining multiple AUROC methods
- **Temporal smoothing**: reducing noise in metric progression
- **Adaptive patience**: dynamic patience adjustment
- **Min epochs protection**: preventing premature stopping
- **Fallback handling**: graceful degradation when ensemble methods unavailable

**Key Test Cases:**
- `test_get_ensemble_metric_basic`: Tests ensemble of multiple AUROC methods
- `test_apply_smoothing`: Verifies temporal smoothing reduces noise
- `test_min_epochs_protection`: Ensures no stopping before minimum epochs

### 📊 Training Visualization (`test_training_visualizer.py`)
Tests for real-time training visualization:
- **Live dashboard generation**: multi-panel training plots
- **Metric tracking**: train/val losses, AUROCs, aggregation comparisons
- **File management**: proper saving of plots and snapshots
- **Update frequency**: controlling visualization update intervals
- **Matplotlib integration**: mocked plotting to avoid display issues

**Key Test Cases:**
- `test_generate_training_dashboard`: Tests complete dashboard creation
- `test_plot_auroc_comparison`: Verifies aggregation method comparison plots
- `test_full_training_simulation`: Integration test for complete training cycle

### 🧠 Enhanced Model (`test_enhanced_supervised_cls.py`)
Tests for enhanced SupervisedClsModel functionality:
- **Individual probability tracking**: per-crop probability storage
- **Enhanced aggregation integration**: automatic application of multiple methods
- **Subject-level AUROC computation**: proper grouping and calculation
- **Enhanced export functionality**: JSON export with method comparisons
- **Backward compatibility**: ensuring standard functionality unchanged

**Key Test Cases:**
- `test_validation_step_individual_tracking`: Verifies individual probability collection
- `test_compute_enhanced_subject_aurocs`: Tests aggregation pipeline
- `test_export_enhanced_subject_results`: Validates enhanced export format

### ⚙️ Enhanced Finetune (`test_enhanced_finetune.py`)
Tests for enhanced CLI and training integration:
- **Argument parsing**: new CLI flags for enhanced features
- **Callback creation**: proper instantiation of enhanced callbacks
- **Model configuration**: passing enhanced settings to model
- **Backward compatibility**: existing arguments unchanged
- **Integration testing**: callbacks and model work together

**Key Test Cases:**
- `test_enhanced_aggregation_argument`: CLI flag for aggregation methods
- `test_create_enhanced_callbacks_all_enabled`: Full callback creation
- `test_backward_compatibility`: Ensures existing functionality preserved

### 📈 Enhanced Analysis (`test_enhanced_analysis.py`)
Tests for post-training analysis tools:
- **Cross-fold analysis**: loading and comparing results across folds
- **Method performance ranking**: identifying best aggregation methods
- **Stability analysis**: coefficient of variation and range metrics
- **Visualization generation**: comprehensive plots and reports
- **Export functionality**: JSON export of detailed results

**Key Test Cases:**
- `test_load_all_fold_results`: Loading enhanced results from multiple folds
- `test_get_method_performance_summary`: Statistical summary across methods
- `test_generate_comprehensive_report`: Complete analysis report generation

## Running Tests

### Run All Enhanced Tests
```bash
# From repository root
cd src
python tests/run_enhanced_tests.py
```

### Run Specific Test Categories
```bash
# Individual categories
python tests/run_enhanced_tests.py aggregation
python tests/run_enhanced_tests.py early_stopping
python tests/run_enhanced_tests.py visualization
python tests/run_enhanced_tests.py model
python tests/run_enhanced_tests.py finetune
python tests/run_enhanced_tests.py analysis
```

### Run Individual Test Files
```bash
# Specific test modules
python -m unittest tests.test_subject_aggregation -v
python -m unittest tests.test_robust_early_stopping -v
python -m unittest tests.test_training_visualizer -v
python -m unittest tests.test_enhanced_supervised_cls -v
python -m unittest tests.test_enhanced_finetune -v
python -m unittest tests.test_enhanced_analysis -v
```

### Quick Interactive Testing
```bash
# Interactive test runner
python tests/quick_test.py
```

## Test Dependencies

Enhanced tests require additional dependencies for mocking and testing:
```bash
pip install matplotlib>=3.5.0 seaborn>=0.11.0 scikit-learn>=1.0.0
```

**Mocking Strategy:**
- `matplotlib` and `seaborn` are mocked to avoid display requirements
- PyTorch Lightning components are mocked for isolated testing
- File I/O uses temporary directories for safe testing

## Expected Test Results

When running the complete test suite, you should see:

```
🧠 FOMO25 Enhanced Features Test Suite
================================================================================

📋 Loading tests from test_subject_aggregation...
   ✅ Loaded 15 tests
📋 Loading tests from test_robust_early_stopping...
   ✅ Loaded 12 tests
📋 Loading tests from test_training_visualizer...
   ✅ Loaded 18 tests
📋 Loading tests from test_enhanced_supervised_cls...
   ✅ Loaded 14 tests
📋 Loading tests from test_enhanced_finetune...
   ✅ Loaded 10 tests
📋 Loading tests from test_enhanced_analysis...
   ✅ Loaded 16 tests

🎯 Total tests loaded: 85

🚀 Running enhanced features tests...
--------------------------------------------------------------------------------

[Individual test results...]

================================================================================
📊 TEST SUMMARY
================================================================================
🎉 ALL TESTS PASSED!
✅ Ran 85 tests successfully
================================================================================
```

## Test Development Guidelines

### Writing New Tests
1. **Follow naming convention**: `test_<functionality>_<specific_case>`
2. **Use descriptive assertions**: Include helpful error messages
3. **Mock external dependencies**: Avoid real file I/O, plotting, network calls
4. **Test edge cases**: Empty data, single items, error conditions
5. **Verify mathematical correctness**: For aggregation and metric computations

### Mock Strategy
```python
# Example mocking pattern
with patch.dict('sys.modules', {
    'matplotlib': MagicMock(),
    'matplotlib.pyplot': MagicMock(),
    'seaborn': MagicMock()
}):
    from utils.training_visualizer import TrainingVisualizer
```

### Temporary Resources
```python
# Use temporary directories for file operations
import tempfile
import shutil

def setUp(self):
    self.temp_dir = tempfile.mkdtemp()

def tearDown(self):
    shutil.rmtree(self.temp_dir)
```

## Integration with CI/CD

These tests are designed to run in automated environments:
- **No display requirements**: All visualization components mocked
- **No external dependencies**: Tests run with standard Python libraries
- **Fast execution**: Most tests complete in milliseconds
- **Comprehensive coverage**: Tests cover all enhanced feature code paths

Add to your CI pipeline:
```yaml
# Example GitHub Actions step
- name: Run Enhanced Features Tests
  run: |
    cd src
    python tests/run_enhanced_tests.py
```

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Ensure src is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**Missing Dependencies:**
```bash
# Install enhanced requirements
pip install -r requirements_enhanced.txt
```

**Display Errors (if mocking fails):**
```bash
# Set headless backend
export MPLBACKEND=Agg
```

### Test-Specific Issues

**Subject Aggregation Tests:**
- Verify numpy and torch installations
- Check mathematical computations for floating-point precision

**Visualization Tests:**
- Ensure matplotlib mocking is working
- Verify temporary directory creation/cleanup

**Model Tests:**
- Mock PyTorch Lightning properly
- Check tensor operations and device compatibility

**Analysis Tests:**
- Verify JSON file operations
- Check pathlib usage for cross-platform compatibility
