# Grammar-Constrained Symbolic Regression for Systematic Alpha Discovery

A methodology for discovering quantitative trading signals (alphas) using symbolic regression constrained by a grammar derived from empirical regularities of financial time series.

## Overview

This project implements the methodology described in the working paper "Grammar-Constrained Symbolic Regression for Systematic Alpha Discovery" by Patryk Kozak (May 2026).

The approach uses:
- **Stylized facts** of financial time series as a prior for feature engineering
- **Symbolic regression** (via PySR) to discover alpha expressions
- **Grammar constraints** to ensure interpretability and prevent overfitting
- **Temporal validation** with strict train/validation/test splits
- **Deterministic code generation** via SymPy for deployment

## Project Structure

```
grammar_constrained_sr/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── loader.py          # Data loading and preprocessing
│   │   ├── panel.py           # Panel data construction
│   │   └── synthetic.py       # Synthetic data generation
│   ├── features/
│   │   ├── base.py           # Base primitive classes
│   │   ├── volatility.py      # Volatility-related primitives
│   │   ├── autocorrelation.py # Autocorrelation primitives
│   │   ├── spectral.py        # Spectral analysis primitives
│   │   ├── wavelet.py         # Wavelet decomposition
│   │   ├── entropy.py         # Entropy-based primitives
│   │   ├── hilbert.py         # Hilbert transform primitives
│   │   ├── hurst.py           # Hurst exponent estimation
│   │   ├── momentum.py        # Momentum primitives
│   │   ├── volume.py          # Volume-related primitives
│   │   ├── leverage.py        # Leverage effect primitives
│   │   ├── skewness.py        # Skewness and asymmetry
│   │   ├── mean_reversion.py  # Mean reversion primitives
│   │   ├── lead_lag.py        # Lead-lag effect primitives
│   │   ├── taylor.py          # Taylor effect primitives
│   │   ├── efficiency.py      # Market efficiency primitives
│   │   └── factory.py         # Feature factory
│   ├── normalization/
│   │   └── cross_sectional.py # Cross-sectional normalization
│   ├── symbolic/
│   │   ├── pysr_config.py    # PySR configuration
│   │   ├── regression.py      # Symbolic regression runner
│   │   └── expressions.py     # Expression handling
│   ├── validation/
│   │   ├── temporal.py        # Temporal validation splits
│   │   ├── statistical.py      # Statistical filtering
│   │   └── metrics.py         # Performance metrics
│   ├── policy/
│   │   ├── base.py           # Base policy classes
│   │   ├── quintile.py        # Quintile long-short policy
│   │   ├── threshold.py       # Threshold-based policies
│   │   ├── kelly.py           # Kelly criterion policy
│   │   └── optimizer.py       # Policy optimization
│   ├── codegen/
│   │   ├── sympy_utils.py     # SymPy utilities
│   │   └── generator.py       # Code generation
│   └── pipeline/
│       ├── main.py           # Main pipeline orchestrator
│       └── workflow.py        # Workflow management
├── tests/
│   ├── test_features.py
│   ├── test_normalization.py
│   ├── test_symbolic.py
│   ├── test_validation.py
│   ├── test_policy.py
│   └── test_codegen.py
├── docs/
│   ├── methodology.md
│   ├── primitives.md
│   ├── validation.md
│   └── deployment.md
├── examples/
│   ├── basic_example.py
│   ├── synthetic_recovery.py
│   └── full_pipeline.py
└── configs/
    ├── pysr_config.yaml
    ├── validation_config.yaml
    └── feature_config.yaml
```

## Installation

```bash
# Clone the repository
git clone https://github.com/datasound-projects/projects-hub.git
cd projects-hub/grammar_constrained_sr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

```
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
sympy>=1.12.0
pysr>=1.2.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
pywavelets>=1.4.0
numba>=0.57.0
optuna>=3.0.0
tqdm>=4.65.0
```

## Quick Start

```python
from src.pipeline.main import AlphaDiscoveryPipeline
from src.data.loader import load_sample_data

# Load sample data
data = load_sample_data()

# Initialize pipeline
pipeline = AlphaDiscoveryPipeline()

# Run full pipeline
results = pipeline.run(data)

# Get discovered alphas
alphas = results['alphas']
```

## Key Features

1. **15 Stylized Fact Primitives**: Comprehensive implementation of all stylized facts
2. **Layered Architecture**: Clear separation of raw data, primitives, and combinations
3. **Grammar Constraints**: Restricted operator set and composition rules
4. **Temporal Validation**: Strict train/validation/test splits
5. **Statistical Filtering**: Multi-criterion validation (IC, Sharpe, stability)
6. **Deterministic Code Generation**: SymPy-based code export
7. **Policy Separation**: Alpha expressions separate from execution policies

## Methodology

The pipeline follows these stages:

1. **Grammar Design**: Define primitives based on stylized facts
2. **Feature Computation**: Compute all primitives for the dataset
3. **Symbolic Regression**: Use PySR to discover alpha expressions
4. **Statistical Filtering**: Validate candidates using multiple criteria
5. **Code Generation**: Export validated alphas as deployable code
6. **Policy Optimization**: Optimize execution rules for validated alphas
7. **Fusion**: Combine alpha and policy for deployment

## Configuration

The pipeline is highly configurable through YAML files in the `configs/` directory:

- `pysr_config.yaml`: PySR parameters (populations, iterations, etc.)
- `validation_config.yaml`: Statistical filter thresholds
- `feature_config.yaml`: Window sizes, primitive parameters

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_features.py

# Run with coverage
pytest --cov=src tests/
```

## Documentation

Full documentation is available in the `docs/` directory:

- [Methodology Overview](docs/methodology.md)
- [Primitive Definitions](docs/primitives.md)
- [Validation Protocol](docs/validation.md)
- [Deployment Guide](docs/deployment.md)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License

## Citation

If you use this work, please cite:

```
Kozak, P. (2026). Grammar-Constrained Symbolic Regression for Systematic Alpha Discovery. 
Working Paper, May 2026.
```
