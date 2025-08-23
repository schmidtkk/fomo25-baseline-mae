# Learning Rate Plotting Enhancement

## Overview

The `LossPlottingCallback` has been enhanced to include learning rate visualization in the training progress plots. This provides better monitoring of learning rate schedules during training.

## Features Added

### Learning Rate Tracking
- **Automatic Collection**: Learning rates are automatically collected from the optimizer every `log_every_n_steps` during training
- **Multi-Optimizer Support**: Uses the first optimizer's first parameter group learning rate
- **Step-based Tracking**: Learning rates are tracked against global training steps for precise visualization

### Enhanced Plotting Layout
- **2x3 Subplot Layout**: Expanded from 2x2 to 2x3 to accommodate learning rate plot
- **Learning Rate Subplot**: Positioned at top-right (axes[0][2]) for high visibility
- **Log Scale Visualization**: Learning rate uses logarithmic scale for better readability
- **Current LR Annotation**: Displays current learning rate value with yellow highlight box

## Plot Layout

```
┌─────────────────┬─────────────────┬─────────────────┐
│ Training Loss   │ Validation Loss │ Learning Rate   │
├─────────────────┼─────────────────┼─────────────────┤
│ Combined Loss   │ AUROC Progress  │ (Reserved)      │
└─────────────────┴─────────────────┴─────────────────┘
```

## Usage

The learning rate plotting is automatically enabled when using `LossPlottingCallback`. No additional configuration is required.

```python
from src.utils.enhanced_callbacks import LossPlottingCallback

# Create callback with learning rate plotting enabled
plotting_callback = LossPlottingCallback(
    plot_every_n_epochs=10,
    log_every_n_steps=50,
    save_dir="training_plots",
    figure_size=(18, 10),  # Increased width for 2x3 layout
)
```

## Learning Rate Visualization Features

1. **Log Scale**: Learning rates are plotted on logarithmic y-axis for better visualization of scheduler behavior
2. **Current Value Display**: Yellow annotation box shows the current learning rate in scientific notation
3. **Step-based X-axis**: X-axis shows global training steps for precise timing correlation with loss curves
4. **Robust Error Handling**: Failed learning rate collection doesn't interrupt training

## Supported Schedulers

Works with all PyTorch learning rate schedulers including:
- `CosineAnnealingLR`
- `ReduceLROnPlateau`
- `StepLR`
- `ExponentialLR`
- Custom schedulers

## Integration with Existing Features

- **Best Metrics Annotations**: Learning rate plot doesn't interfere with existing best metrics tracking
- **Smoothing**: Learning rate data is not smoothed (shows actual scheduler values)
- **File Output**: Same `training_progress_latest.png` filename with expanded layout

## Configuration

The default figure size has been updated to (18, 10) to accommodate the 2x3 layout, but can be customized:

```python
plotting_callback = LossPlottingCallback(
    figure_size=(21, 12),  # Custom size for larger displays
)
```

## Troubleshooting

### Learning Rate Not Displayed
- Check that optimizer is properly configured in your Lightning module
- Verify that `configure_optimizers()` returns the optimizer correctly
- Ensure training steps are being logged (check `log_every_n_steps` parameter)

### Plot Layout Issues
- Use figure_size=(18, 10) or larger for optimal 2x3 layout visibility
- Adjust `figure_size` parameter if plots appear cramped

## Example Output

The learning rate subplot shows:
- Magenta line plotting learning rate vs. global steps
- Logarithmic y-axis scale
- Current LR annotation in yellow box
- Grid for easier reading
- "Learning Rate Schedule" title

This enhancement provides comprehensive monitoring of training dynamics by visualizing learning rate changes alongside loss curves and performance metrics.
