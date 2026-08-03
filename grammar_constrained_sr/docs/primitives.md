# Stylized Fact Primitives

This document provides detailed definitions of all primitives implemented in the methodology, organized by stylized fact.

## Overview

Each primitive:
- Takes a (T × N) panel of data as input
- Computes a feature that captures a specific stylized fact
- Returns a (T × N) panel of feature values
- Is numerically stable and produces bounded outputs where possible
- Has a clear economic interpretation

## Primitive Categories

### 1. Fat Tails (Stylized Fact 1)

#### RollingKurtosis
- **Definition**: Rolling fourth standardized moment (excess kurtosis)
- **Formula**: E[(X - μ)⁴ / σ⁴] - 3
- **Interpretation**: >0 indicates fatter tails than normal distribution
- **Window**: 5, 10, 20, 60 days
- **Bounds**: Typically [-∞, ∞], but practically bounded

#### HillEstimator
- **Definition**: Power-law tail index estimator
- **Formula**: 1 / (mean(log(X_i / X_min))) for X_i > threshold
- **Interpretation**: Lower values indicate fatter tails, higher extreme risk
- **Window**: 20, 60 days
- **Threshold**: Top 5% of observations
- **Bounds**: (0, ∞)

#### ExtremeRatio
- **Definition**: Ratio of extreme observations to normal expectation
- **Formula**: (count(|X| > 2σ) / N) / (2 * (1 - Φ(2))) where Φ is CDF
- **Interpretation**: >1 confirms fat-tailed behavior
- **Window**: 20, 60 days
- **Threshold**: 2 standard deviations
- **Bounds**: [0, ∞)

### 2. Volatility Clustering (Stylized Fact 2)

#### RollingVolatility
- **Definition**: Rolling standard deviation of returns
- **Formula**: sqrt(E[(X - μ)²])
- **Interpretation**: Measure of return dispersion
- **Window**: 5, 10, 20, 60 days
- **Bounds**: [0, ∞)

#### VolatilityClustering
- **Definition**: Autocorrelation of absolute returns
- **Formula**: corr(|X_t|, |X_{t+lag}|)
- **Interpretation**: >0 indicates volatility clustering (ARCH effect)
- **Window**: 20, 60 days
- **Lag**: 1, 5, 10 days
- **Bounds**: [-1, 1]

#### VolatilityRatio
- **Definition**: Ratio of short-term to long-term volatility
- **Formula**: σ_short / σ_long
- **Interpretation**: >1 indicates current high-volatility regime
- **Short window**: 5 days
- **Long window**: 60 days
- **Bounds**: [0, ∞)

#### EWMVolatility
- **Definition**: Exponentially weighted moving average volatility
- **Formula**: EWMA of |X_t|
- **Interpretation**: Faster-reacting volatility estimate
- **Half-life**: 10, 30, 90 days
- **Bounds**: [0, ∞)

#### VolatilityOfVolatility
- **Definition**: Standard deviation of rolling volatility
- **Formula**: std(σ_t) over outer window
- **Interpretation**: Measures volatility stability
- **Inner window**: 20 days
- **Outer window**: 60 days
- **Bounds**: [0, ∞)

### 3. Leverage Effect (Stylized Fact 3)

#### DownsideUpsideVolatilityRatio
- **Definition**: Ratio of downside to upside volatility
- **Formula**: σ_down / σ_up where σ_down = std(X_t | X_t < 0), σ_up = std(X_t | X_t > 0)
- **Interpretation**: >1 indicates leverage effect (negative returns increase volatility more)
- **Window**: 20, 60 days
- **Bounds**: [0, ∞)

#### LeverageCorrelation
- **Definition**: Correlation between returns and volatility changes
- **Formula**: corr(X_t, Δσ_t)
- **Interpretation**: <0 indicates leverage effect
- **Window**: 20, 60 days
- **Volatility window**: 20 days
- **Bounds**: [-1, 1]

#### GJRAsymmetry
- **Definition**: GJR-GARCH inspired asymmetry measure
- **Formula**: E[X_t² | X_t < 0] / E[X_t² | X_t > 0]
- **Interpretation**: >1 indicates leverage effect
- **Window**: 20, 60 days
- **Bounds**: [0, ∞)

### 4. Volume-Volatility Correlation (Stylized Fact 4)

#### VolumeVolatilityCorrelation
- **Definition**: Correlation between absolute returns and volume
- **Formula**: corr(|X_t|, V_t) where V_t is volume
- **Interpretation**: >0 indicates volume increases with volatility
- **Window**: 20, 60 days
- **Bounds**: [-1, 1]

#### RelativeVolume
- **Definition**: Current volume relative to rolling average
- **Formula**: V_t / E[V_t]
- **Interpretation**: >1 indicates above-average volume
- **Window**: 20, 60 days
- **Bounds**: [0, ∞)

#### VolumeSurprise
- **Definition**: Current volume relative to rolling maximum
- **Formula**: V_t / max(V_{t-w:t})
- **Interpretation**: [0, 1] indicates volume as fraction of recent maximum
- **Window**: 20, 60 days
- **Bounds**: [0, 1]

#### VolumeWeightedVolatility
- **Definition**: Volatility weighted by trading volume
- **Formula**: std(V_t * X_t)
- **Interpretation**: Volume-weighted measure of "true" volatility
- **Window**: 20, 60 days
- **Bounds**: [0, ∞)

#### DollarVolume
- **Definition**: Volume multiplied by typical price
- **Formula**: V_t * (H_t + L_t + C_t) / 3
- **Interpretation**: Monetary value of trading activity
- **Bounds**: [0, ∞)

### 5. Absence of Linear Autocorrelation (Stylized Fact 5)

#### RollingAutocorrelation
- **Definition**: Rolling first-order autocorrelation of returns
- **Formula**: corr(X_t, X_{t+1})
- **Interpretation**: ≈0 indicates weak-form efficiency
- **Window**: 20, 60 days
- **Lag**: 1
- **Bounds**: [-1, 1]

#### VarianceRatio
- **Definition**: Lo-MacKinlay variance ratio
- **Formula**: var(X_{t:t+k}) / (k * var(X_t))
- **Interpretation**: >1 suggests momentum, <1 suggests mean reversion
- **Window**: 20, 60 days
- **k**: 2, 5, 10
- **Bounds**: [0, ∞)

### 6. Slow Decay of Autocorrelation in Absolute Returns (Stylized Fact 6)

#### HurstExponent (R/S Method)
- **Definition**: Hurst exponent via Rescaled Range method
- **Formula**: H = log(R/S) / log(n) + 0.5
- **Interpretation**: >0.5 persistence, <0.5 antipersistence, =0.5 random walk
- **Window**: 20, 60, 120 days
- **Bounds**: [0, 1]

#### HurstExponent (DFA Method)
- **Definition**: Hurst exponent via Detrended Fluctuation Analysis
- **Formula**: Slope of log(F(n)) vs log(n) where F(n) is fluctuation function
- **Interpretation**: >0.5 persistence, <0.5 antipersistence
- **Window**: 20, 60, 120 days
- **Bounds**: [0, 1]

#### AutocorrelationDecay
- **Definition**: Decay exponent of absolute return autocorrelation
- **Formula**: Fit power law: ACF(lag) = a * lag^(-b), return b
- **Interpretation**: Measures speed of ACF decay
- **Window**: 20, 60 days
- **Max lag**: 10
- **Bounds**: [0, ∞)

#### ACFDecayRatio
- **Definition**: Ratio of ACF at long lag to ACF at short lag
- **Formula**: ACF(lag_long) / ACF(lag_short)
- **Interpretation**: <1 indicates slow decay (persistence)
- **Short lag**: 1
- **Long lag**: 10
- **Window**: 20, 60 days
- **Bounds**: [-∞, ∞]

### 7. Negative Skewness (Stylized Fact 7)

#### RollingSkewness
- **Definition**: Rolling third standardized moment
- **Formula**: E[(X - μ)³ / σ³]
- **Interpretation**: <0 indicates negative skewness (fat left tail)
- **Window**: 20, 60 days
- **Bounds**: [-∞, ∞]

#### DownsideFrequency
- **Definition**: Ratio of negative to positive extreme returns
- **Formula**: count(X < -threshold) / count(X > threshold)
- **Interpretation**: >1 indicates more frequent large negative returns
- **Window**: 20, 60 days
- **Threshold**: 1%, 2% return
- **Bounds**: [0, ∞)

### 8. Aggregational Gaussianity (Stylized Fact 8)

#### KurtosisRatio
- **Definition**: Ratio of daily to aggregated return kurtosis
- **Formula**: kurtosis(X_daily) / kurtosis(X_aggregated)
- **Interpretation**: >1 indicates non-Gaussianity concentrated at high frequencies
- **Daily window**: 20 days
- **Aggregated window**: 100 days
- **Aggregation period**: 5 days
- **Bounds**: [-∞, ∞]

### 9. Gain/Loss Asymmetry (Stylized Fact 9)

#### UpDownRunRatio
- **Definition**: Ratio of average up run length to down run length
- **Formula**: E[L_up] / E[L_down] where L is run length
- **Interpretation**: >1 indicates gains develop slowly, losses arrive abruptly
- **Window**: 20, 60 days
- **Bounds**: [0, ∞)

#### ConditionalMeanRatio
- **Definition**: Ratio of mean positive to mean negative return
- **Formula**: E[X | X > 0] / |E[X | X < 0]|
- **Interpretation**: >1 indicates larger positive than negative returns
- **Window**: 20, 60 days
- **Bounds**: [0, ∞)

#### GainLossAsymmetry
- **Definition**: Combined measure of gain/loss asymmetry
- **Formula**: (skewness + (1 - mean_ratio)) / 2
- **Interpretation**: Higher values indicate more asymmetry
- **Window**: 20, 60 days
- **Bounds**: [-∞, ∞]

### 10. Mean Reversion (Stylized Fact 10)

#### HodrickPrescottCycle
- **Definition**: Deviation from HP filter trend
- **Formula**: X_t - τ_t where τ_t is HP trend
- **Interpretation**: Positive = above trend (potential mean reversion)
- **Window**: 60, 120 days
- **Lambda**: 1600 (standard for daily data)
- **Bounds**: [-∞, ∞]

#### HilbertAmplitude
- **Definition**: Instantaneous amplitude from Hilbert transform
- **Formula**: |analytic signal| = sqrt(X_t² + H[X_t]²)
- **Interpretation**: Strength of current cycle
- **Window**: 20, 60 days
- **Bounds**: [0, ∞)

#### HilbertPhase
- **Definition**: Instantaneous phase from Hilbert transform
- **Formula**: angle(analytic signal) = atan2(H[X_t], X_t)
- **Interpretation**: Position in cycle (0=through, π/2=ascending, π=peak, -π/2=descending)
- **Window**: 20, 60 days
- **Bounds**: [-π, π]

#### ZeroCrossingRate
- **Definition**: Rate of sign changes in demeaned returns
- **Formula**: count(sign(X_t - μ) ≠ sign(X_{t-1} - μ)) / (n - 1)
- **Interpretation**: Higher values indicate more mean-reverting behavior
- **Window**: 20, 60 days
- **Bounds**: [0, 1]

#### NormalizedDeviation
- **Definition**: Z-score of returns relative to rolling mean
- **Formula**: (X_t - μ) / σ
- **Interpretation**: How far current return is from local average
- **Window**: 20, 60 days
- **Bounds**: [-∞, ∞]

#### MeanReversionSignal
- **Definition**: Combined mean reversion signal
- **Formula**: (z_score + zero_crossing_rate) / 2
- **Interpretation**: Higher values indicate stronger mean reversion signal
- **Window**: 20, 60 days
- **Bounds**: [-∞, ∞]

### 11. Momentum (Stylized Fact 11)

#### RollingReturn
- **Definition**: Cumulative return over rolling window
- **Formula**: Π(1 + X_t) - 1
- **Interpretation**: Total return over the window
- **Window**: 20, 60, 120, 252 days
- **Bounds**: [-∞, ∞]

#### CumulativeReturn
- **Definition**: Cumulative returns over multiple windows
- **Formula**: Π(1 + X_t) - 1 for each window
- **Interpretation**: Returns over different horizons
- **Windows**: 20, 60, 120, 252 days
- **Bounds**: [-∞, ∞]

#### TrendStrength
- **Definition**: Rolling Sharpe ratio
- **Formula**: (μ / σ) * sqrt(252 / window)
- **Interpretation**: Strength of trend relative to volatility
- **Window**: 60, 120 days
- **Bounds**: [-∞, ∞]

#### TrendLinearity
- **Definition**: R² of linear fit to cumulative returns
- **Formula**: 1 - SS_res / SS_tot
- **Interpretation**: How clean the trend is (0=no trend, 1=perfect trend)
- **Window**: 60, 120 days
- **Bounds**: [0, 1]

#### JegadeeshTitmanMomentum
- **Definition**: 12-month return minus most recent month
- **Formula**: R_{t-12:t-1} - R_{t-1:t}
- **Interpretation**: Momentum avoiding short-term reversal
- **Long window**: 252 days
- **Short window**: 20 days
- **Bounds**: [-∞, ∞]

### 12. Lead-Lag Effects (Stylized Fact 12)

#### RollingBeta
- **Definition**: Rolling beta to market
- **Formula**: cov(X, M) / var(M) where M is market return
- **Interpretation**: Sensitivity to market movements
- **Window**: 60, 120 days
- **Bounds**: [-∞, ∞]

#### HouMoskowitzDelay
- **Definition**: Additional explanatory power from lagged market
- **Formula**: R²(lagged M) - R²(contemporaneous M)
- **Interpretation**: >0 indicates asset reacts slowly to market information
- **Window**: 60 days
- **Lag**: 1 day
- **Bounds**: [-1, 1]

#### LeadLagEffect
- **Definition**: Combined lead-lag measure
- **Formula**: beta - delay
- **Interpretation**: Higher = more leader, lower = more laggard
- **Window**: 60 days
- **Bounds**: [-∞, ∞]

### 13. Coarse-Fine Volatility Asymmetry (Stylized Fact 13)

#### WaveletEnergy
- **Definition**: Variance of wavelet detail coefficients
- **Formula**: var(detail_coeffs) for each decomposition level
- **Interpretation**: Energy at different time scales
- **Wavelet**: Daubechies 4
- **Levels**: 1, 2, 3 (≈2-day, 4-day, 8-day fluctuations)
- **Bounds**: [0, ∞)

#### WaveletNoiseToSignalRatio
- **Definition**: Ratio of high-frequency to low-frequency energy
- **Formula**: energy(level_1) / energy(level_3)
- **Interpretation**: >1 indicates more noise than signal
- **Wavelet**: Daubechies 4
- **Levels**: 1 (noise), 3 (signal)
- **Bounds**: [0, ∞)

#### SpectralPowerRatio
- **Definition**: Ratio of low-frequency to high-frequency energy
- **Formula**: sum(PSD[low]) / sum(PSD[high])
- **Interpretation**: >1 indicates more power in low frequencies
- **NFFT**: 256
- **Low frequency ratio**: 0.25
- **Bounds**: [0, ∞)

### 14. Taylor Effect (Stylized Fact 14)

#### TaylorRatio
- **Definition**: Ratio of ACF(|r|) to ACF(r²)
- **Formula**: corr(|X_t|, |X_{t+1}|) / corr(X_t², X_{t+1}²)
- **Interpretation**: ≈1 indicates Taylor effect (d=1 is optimal)
- **Lag**: 1
- **Window**: 20, 60 days
- **Bounds**: [-∞, ∞]

#### OptimalTaylorExponent
- **Definition**: Value of d that maximizes ACF(|r|^d)
- **Formula**: argmax_d corr(|X_t|^d, |X_{t+1}|^d)
- **Interpretation**: ≈1 confirms Taylor effect
- **d range**: [0.1, 2.0]
- **d steps**: 20
- **Window**: 20, 60 days
- **Bounds**: [0.1, 2.0]

### 15. Time-Varying Market Efficiency (Stylized Fact 15)

#### SpectralEntropy
- **Definition**: Shannon entropy of normalized power spectral density
- **Formula**: -Σ p_i log(p_i) where p_i = PSD_i / Σ PSD
- **Interpretation**: Lower = more concentrated spectral power = more predictable = less efficient
- **NFFT**: 256
- **Window**: 20, 60 days
- **Bounds**: [0, log(NFFT)]

#### PermutationEntropy
- **Definition**: Ordinal-pattern-based complexity measure
- **Formula**: Shannon entropy of permutation patterns
- **Interpretation**: Lower = more regular patterns = less efficient
- **m (embedding dimension)**: 3
- **delay**: 1
- **Window**: 20, 60 days
- **Bounds**: [0, log(m!)]

#### AR1RSquared
- **Definition**: R² of rolling AR(1) model
- **Formula**: corr(X_t, X_{t-1})²
- **Interpretation**: Higher = more predictable = less efficient
- **Window**: 20, 60 days
- **Bounds**: [0, 1]

#### MarketEfficiency
- **Definition**: Combined market efficiency measure
- **Formula**: (1 - spectral_entropy + 1 - perm_entropy + ar1_rsquared) / 3
- **Interpretation**: Higher = less efficient market
- **Window**: 20, 60 days
- **Bounds**: [0, 1]

#### AdaptiveMarketHypothesis
- **Definition**: AMH measure of time-varying efficiency
- **Formula**: (1 - spectral_entropy + ar1_rsquared + volatility) / 3
- **Interpretation**: Higher = less efficient (more predictable)
- **Window**: 20, 60 days
- **Bounds**: [0, 1]

## Normalization Methods

All primitives are passed through **cross-sectional normalization** to make them comparable across assets:

### Percentile Rank Normalization
- **Formula**: rank(x) = (number of values ≤ x) / (total number of values)
- **Output**: [0, 1]
- **Interpretation**: 0 = lowest value, 1 = highest value

### Z-Score Normalization
- **Formula**: z(x) = (x - μ) / σ
- **Output**: (-∞, ∞), typically [-3, 3] after clipping
- **Interpretation**: 0 = mean, ±1 = ±1 standard deviation

### Min-Max Normalization
- **Formula**: minmax(x) = (x - min) / (max - min)
- **Output**: [0, 1]
- **Interpretation**: 0 = minimum, 1 = maximum

## Feature Naming Convention

Features follow a consistent naming convention:

```
{primitive_name}_w{window}_{normalization}
```

Examples:
- `rolling_volatility_w20_rank` - Rank-normalized 20-day rolling volatility
- `hurst_exponent_w60_zscore` - Z-score normalized 60-day Hurst exponent
- `hilbert_amplitude_w20_raw` - Raw 20-day Hilbert amplitude

## Implementation Notes

1. **Rolling Windows**: All primitives use rolling windows to handle non-stationarity
2. **NaN Handling**: First `window-1` observations are NaN (insufficient data)
3. **Numerical Stability**: Division by zero is avoided, log(0) is handled
4. **Performance**: Some primitives (Hurst, wavelet) are computationally expensive
5. **Dependencies**: Wavelet primitives require `pywavelets` package

## Primitive Selection Guidelines

When adding new primitives:

1. **Economic Motivation**: Must correspond to a documented stylized fact
2. **Numerical Stability**: Must handle edge cases (zero division, NaN, etc.)
3. **Bounded Output**: Should produce bounded outputs where possible
4. **Interpretability**: Must have clear economic interpretation
5. **Computational Efficiency**: Should be reasonably fast to compute
6. **Distinct Information**: Should capture information not already represented

## Primitive Validation

Each primitive is validated with:

1. **Unit Tests**: Basic functionality and edge cases
2. **Synthetic Data**: Known patterns to verify correct computation
3. **Real Data**: Empirical validation on financial time series
4. **Economic Interpretation**: Manual review of economic meaning
