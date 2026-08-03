# Methodology Overview

This document provides a detailed overview of the Grammar-Constrained Symbolic Regression methodology for Systematic Alpha Discovery.

## Core Concept

The methodology combines **symbolic regression** with **financial stylized facts** to create a principled approach to alpha discovery that:

1. **Reduces the search space** by constraining the symbolic regression to econometrically motivated primitives
2. **Improves interpretability** by ensuring every expression component has economic meaning
3. **Prevents overfitting** through structural constraints and rigorous validation

## Pipeline Architecture

The pipeline follows a **layered architecture** with clear separation of concerns:

```
Layer 0: Raw Market Data
├── OHLCV (Open, High, Low, Close, Volume)
└── Derived: log returns, forward returns, typical price, dollar volume

Layer 1: Signal Processing Primitives
├── Volatility: rolling std, EWMA, volatility clustering, leverage effect
├── Autocorrelation: ACF, variance ratio, Hurst exponent
├── Spectral: spectral entropy, power ratio, permutation entropy
├── Wavelet: energy at different scales, noise-to-signal ratio
├── Hilbert: amplitude, phase
├── Momentum: rolling returns, trend strength, linearity
├── Volume: volume-volatility correlation, relative volume
├── Skewness: rolling skewness, downside frequency
├── Mean Reversion: HP cycle, zero-crossing rate, normalized deviation
├── Lead-Lag: rolling beta, Hou-Moskowitz delay
├── Taylor Effect: Taylor ratio, optimal exponent
└── Efficiency: AR(1) R-squared, market efficiency measures

Layer 2: Cross-Sectional Normalization
├── Raw features (optional)
├── Percentile rank normalization [0, 1]
└── Z-score normalization (mean=0, std=1)

Layer 3: Symbolic Regression Search Space
├── Binary operators: +, -, ×, ÷
├── Unary operators: abs, neg
├── Composition rules: no nesting of signal processing primitives
└── Complexity constraints: max size, parsimony pressure
```

## The 15 Stylized Facts

The methodology is built on 15 well-documented empirical regularities of financial time series:

### 1. Fat Tails (Excess Kurtosis)
- **Primitive**: Rolling kurtosis, Hill estimator, extreme ratio
- **Interpretation**: Extreme returns occur more frequently than predicted by Gaussian distribution
- **Economic significance**: Higher tail risk, need for robust risk management

### 2. Volatility Clustering (ARCH Effect)
- **Primitive**: Autocorrelation of absolute returns, volatility ratio, EWMA volatility, volatility of volatility
- **Interpretation**: Large price movements tend to be followed by large movements
- **Economic significance**: Time-varying risk, regime detection

### 3. Leverage Effect
- **Primitive**: Downside-upside volatility ratio, leverage correlation, GJR asymmetry
- **Interpretation**: Negative returns increase future volatility more than positive returns
- **Economic significance**: Asymmetric risk, options pricing implications

### 4. Volume-Volatility Correlation
- **Primitive**: Volume-volatility correlation, relative volume, volume surprise, volume-weighted volatility
- **Interpretation**: Trading volume increases with absolute returns
- **Economic significance**: Information arrival, liquidity provision

### 5. Absence of Linear Autocorrelation
- **Primitive**: Rolling autocorrelation, variance ratio
- **Interpretation**: Returns exhibit near-zero autocorrelation
- **Economic significance**: Weak-form market efficiency

### 6. Slow Decay of Autocorrelation in Absolute Returns
- **Primitive**: Hurst exponent, autocorrelation decay, ACF decay ratio
- **Interpretation**: Absolute returns exhibit long-range dependence
- **Economic significance**: Persistence, memory in volatility

### 7. Negative Skewness
- **Primitive**: Rolling skewness, downside frequency
- **Interpretation**: Large downward movements more frequent than upward
- **Economic significance**: Crash risk, tail risk asymmetry

### 8. Aggregational Gaussianity
- **Primitive**: Kurtosis ratio
- **Interpretation**: Return distribution converges to Gaussian as aggregation period increases
- **Economic significance**: Microstructure effects, high-frequency anomalies

### 9. Gain/Loss Asymmetry
- **Primitive**: Up-down run ratio, conditional mean ratio
- **Interpretation**: Market drawdowns steep and rapid, recoveries gradual
- **Economic significance**: Behavioral biases, market psychology

### 10. Mean Reversion
- **Primitive**: Hodrick-Prescott cycle, Hilbert amplitude/phase, zero-crossing rate, normalized deviation
- **Interpretation**: At short horizons, returns tend to revert
- **Economic significance**: Short-term trading opportunities

### 11. Momentum
- **Primitive**: Rolling return, cumulative return, trend strength, trend linearity, Jegadeesh-Titman momentum
- **Interpretation**: At intermediate horizons, assets that performed well continue to perform well
- **Economic significance**: Intermediate-term trading strategies

### 12. Lead-Lag Effects
- **Primitive**: Rolling beta, Hou-Moskowitz delay
- **Interpretation**: Large, liquid assets react to information faster than small, illiquid ones
- **Economic significance**: Cross-sectional predictability

### 13. Coarse-Fine Volatility Asymmetry
- **Primitive**: Wavelet energy, wavelet noise-to-signal ratio, spectral power ratio
- **Interpretation**: Low-frequency volatility predicts high-frequency volatility better than reverse
- **Economic significance**: Multi-scale volatility dynamics

### 14. Taylor Effect
- **Primitive**: Taylor ratio, optimal Taylor exponent
- **Interpretation**: Autocorrelation of |r|^d is maximized at d≈1, not d=2
- **Economic significance**: Nonlinear dependence structure

### 15. Time-Varying Market Efficiency
- **Primitive**: Spectral entropy, permutation entropy, AR(1) R-squared, Adaptive Market Hypothesis measure
- **Interpretation**: Market efficiency fluctuates over time
- **Economic significance**: Regime-dependent predictability

## Grammar Constraints

The symbolic regression is constrained by a **grammar** that:

1. **Restricts operators** to basic algebraic operations: {+, -, ×, ÷, abs, neg}
2. **Prohibits nesting** of signal processing primitives (e.g., hilbert(wavelet(...)) is forbidden)
3. **Limits complexity** through maximum expression size and parsimony pressure
4. **Uses only pre-computed primitives** as atomic building blocks

### Why Grammar Constraints Work

1. **Search Space Reduction**: 
   - Unconstrained: Billions of possible expressions
   - Constrained: ~60-80 primitives × 6 operators × depth 4-5 = Orders of magnitude smaller

2. **Built-in Interpretability**:
   - Every primitive has documented economic interpretation
   - Every expression is a composition of known phenomena
   - Enables economic validation before backtesting

3. **Structural Overfitting Protection**:
   - Primitives have no fitted parameters
   - Only expression structure and coefficients are optimized
   - Combined with temporal validation and statistical filtering

## Validation Protocol

The methodology uses a **three-stage temporal validation** protocol:

### Stage 1: Temporal Split
- Train: 60% (first chronologically)
- Validation: 20% (middle)
- Test: 20% (last)
- **No randomization, no shuffling, no overlap**

### Stage 2: Statistical Filter
All candidates must pass **five criteria** on the validation set:

1. **Information Coefficient (IC)**: Spearman correlation > 0.015
2. **IC Stability**: Fraction of positive daily ICs > 55%
3. **Sharpe Ratio**: Annualized Sharpe of quintile long-short > 0.3
4. **Turnover**: Daily turnover < 60%
5. **Complexity**: Expression size < 20 nodes

### Stage 3: Out-of-Sample Evaluation
- Passing candidates evaluated on test set
- Complexity-performance analysis to verify grammar calibration
- Only candidates with positive out-of-sample IC are promoted

## Alpha-Policy Separation

A key innovation is the **separation between alpha expression and policy function**:

### Alpha Expression
- Mathematical function that takes features as input
- Produces a score for each asset (relative attractiveness)
- Says **nothing** about position sizing, timing, or execution
- Pure prediction: ranking of assets by expected future performance

### Policy Function
- Takes alpha scores as input
- Produces trading actions (position sizes, direction)
- Handles all execution dimensions:
  - Direction (long, short, flat)
  - Size (capital allocation)
  - Entry/exit rules
  - Holding duration

### Why Separation Matters

1. **Overfitting Isolation**: 
   - Alpha can overfit to return patterns
   - Policy can overfit to equity curve patterns
   - Separate optimization prevents interaction

2. **Economic Interpretability**:
   - Alpha interpretation independent of execution
   - Can evaluate economic plausibility without execution details

3. **Modularity and Reuse**:
   - Same alpha with different policies for different contexts
   - Same policy with different alphas
   - Independent maintenance and testing

## Synthetic Recovery Test

Before applying to real data, the grammar is validated with a **synthetic recovery test**:

1. Generate synthetic panel with planted alpha signal
2. Compute all primitives
3. Run PySR
4. Check if recovered expression correlates with true signal

**Success criterion**: Spearman correlation > 0.7 for moderate signal strength

This test ensures the grammar has the expressive power to represent known signals.

## Code Generation

Every discovered alpha can be **deterministically translated** to deployable code:

1. **SymPy Export**: Expressions exported as SymPy symbolic objects
2. **Code Generation**: SymPy → Python, NumPy, or C code
3. **Deterministic**: Same expression always produces same code
4. **Transparent**: Code is the alpha, no hidden state or parameters

### Benefits of Deterministic Code Generation

- Version control, auditing, unit testing
- No model serialization needed
- No framework dependencies
- Behavior consistent across environments
- Most transparent form of trading signal

## Policy Optimization

For validated alphas, policy functions are optimized to maximize **growth rate** (Kelly criterion):

1. **Objective**: Maximize expected growth rate of equity curve
2. **Methods**:
   - Parametric grid search (simple, safe)
   - Symbolic regression for policy discovery (powerful, needs careful validation)
   - Reinforcement learning (advanced, state-dependent)

3. **Key Insight**: Growth rate maximization automatically balances return and risk

## Fusion

The final stage combines validated alpha with optimized policy:

```
Trading Rule = Policy(Alpha(Features))
```

**Hard gate**: Fusion only allowed when alpha has independently demonstrated predictive value.

## Extensions

The methodology can be extended with:

1. **Additional Primitives**: From other domains (TDA, information theory, dynamical systems)
2. **Cross-Asset Primitives**: Sector-relative momentum, pairwise correlations
3. **Regime Detection**: Use stylized facts as conditioning variables
4. **Hybrid Approaches**: Symbolic alpha + learned policy (transformer-based)

## Key Advantages

1. **Systematic**: Automated discovery of novel combinations
2. **Interpretable**: Every component has economic meaning
3. **Robust**: Structural protection against overfitting
4. **Deployable**: Deterministic code generation
5. **Modular**: Separation of alpha and policy
6. **Reproducible**: Deterministic mode, version control

## Limitations

1. **Grammar Scope**: Only expresses known regularities
2. **No Transaction Costs**: Backtest is gross of costs
3. **Stylized Fact Stability**: Assumes regularities persist over time
4. **Computational Cost**: Primitive computation can be expensive
5. **Parameter Optimization**: Limited to expression structure, not primitive parameters

## Best Practices

1. **Start Small**: Begin with a subset of stylized facts
2. **Validate Grammar**: Always run synthetic recovery test first
3. **Temporal Validation**: Never use random splits for time series
4. **Multi-Criterion Filtering**: Don't rely on a single metric
5. **Regular Re-runs**: Re-run pipeline periodically to detect alpha decay
6. **Cost Modeling**: Always evaluate net of realistic transaction costs
7. **Risk Management**: Size positions appropriately, use stop-losses
8. **Diversity**: Combine multiple uncorrelated alphas
