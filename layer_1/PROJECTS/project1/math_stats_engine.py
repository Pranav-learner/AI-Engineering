"""
# THE MATHEMATICS WE ARE IMPLEMENTING

Before building Machine Learning models, we need statistical foundations to understand:

* Cash-flow behavior
* Distributions and variability
* Financial volatility
* Relationships between variables
* Anomalies and outliers
* Statistical uncertainty

We will implement these concepts using **NumPy and SciPy**.

---

# 1. CASH-FLOW VECTORS AND BALANCE TRAJECTORY

Daily income:

**i = [i₁, i₂, ..., iₜ]**

Daily expenses:

**e = [e₁, e₂, ..., eₜ]**

Net cash flow:

**c = i − e**

For each day:

**cₜ = iₜ − eₜ**

If the initial balance is **B₀**, then:

**Bₜ = B₀ + ∑ₖ₌₁ᵗ cₖ**

Therefore:

> Balance = Initial Balance + Cumulative Net Cash Flow

---

# 2. MEAN AND SAMPLE STANDARD DEVIATION

For observations:

**x₁, x₂, ..., xₙ**

Mean:

**x̄ = (1/n) × ∑ᵢ₌₁ⁿ xᵢ**

Sample standard deviation:

**s = √[(1/(n−1)) × ∑ᵢ₌₁ⁿ (xᵢ − x̄)²]**

Standard deviation measures how much data varies around its mean.

---

# 3. EXPENSE VOLATILITY — COEFFICIENT OF VARIATION

Standard deviation depends on the scale of the data.

Therefore, we use the Coefficient of Variation:

**CV = s / |x̄|**

Interpretation:

**CV < 0.15** → Low volatility / predictable spending

**0.15 ≤ CV ≤ 0.35** → Moderate volatility

**CV > 0.35** → High volatility / erratic spending

Note: CV becomes unreliable when the mean is close to zero.

---

# 4. PEARSON CORRELATION

For variables:

**X = [x₁, x₂, ..., xₙ]**

**Y = [y₁, y₂, ..., yₙ]**

Pearson correlation:

**r = ∑(xᵢ−x̄)(yᵢ−ȳ) / [√∑(xᵢ−x̄)² × √∑(yᵢ−ȳ)²]**

Range:

**−1 ≤ r ≤ +1**

**r ≈ +1** → Strong positive relationship

**r ≈ −1** → Strong negative relationship

**r ≈ 0** → Little or no linear relationship

Correlation does not imply causation.

---

# 5. COVARIANCE AND COVARIANCE MATRIX

Sample covariance:

**Cov(X,Y) = (1/(n−1)) × ∑(xᵢ−x̄)(yᵢ−ȳ)**

**Cov(X,Y) > 0** → Variables tend to move together

**Cov(X,Y) < 0** → Variables tend to move in opposite directions

For multiple expense categories, we construct a covariance matrix containing:

**Variance on the diagonal**

**Covariance between categories off the diagonal**

We will also construct a correlation matrix.

---

# 6. OUTLIER DETECTION — STANDARD Z-SCORE

Z-score measures how many standard deviations a value is from the mean.

**zᵢ = (xᵢ − x̄) / s**

Outlier rule:

**|zᵢ| > 3 → Potential Outlier**

Limitation: Extreme outliers can distort the mean and standard deviation.

---

# 7. OUTLIER DETECTION — MODIFIED Z-SCORE (MAD)

Median:

**x̃ = median(x)**

Median Absolute Deviation:

**MAD = median(|xᵢ − x̃|)**

Modified Z-score:

**Mᵢ = [0.6745 × (xᵢ − x̃)] / MAD**

Outlier rule:

**|Mᵢ| > 3.5 → Potential Outlier**

This method is more robust against extreme values.

---

# 8. OUTLIER DETECTION — TUKEY'S IQR METHOD

First Quartile:

**Q₁ = 25th percentile**

Third Quartile:

**Q₃ = 75th percentile**

Interquartile Range:

**IQR = Q₃ − Q₁**

Lower Bound:

**Lower = Q₁ − 1.5 × IQR**

Upper Bound:

**Upper = Q₃ + 1.5 × IQR**

Outlier if:

**xᵢ < Lower**

OR

**xᵢ > Upper**

---

# 9. CONFIDENCE INTERVALS FOR MONTHLY EXPENSES

Sample mean:

**x̄ = (1/n) × ∑ᵢ₌₁ⁿ xᵢ**

Sample standard deviation:

**s = √[(1/(n−1)) × ∑ᵢ₌₁ⁿ (xᵢ−x̄)²]**

Standard Error:

**SE = s / √n**

For confidence level:

**100(1−α)%**

Degrees of freedom:

**df = n−1**

The Student's t critical value is:

**t₍₁₋α⁄₂, n−1₎**

Confidence Interval:

**CI = x̄ ± t₍₁₋α⁄₂, n−1₎ × (s/√n)**

Lower Confidence Bound:

**Lower = x̄ − t₍₁₋α⁄₂, n−1₎ × (s/√n)**

Upper Confidence Bound:

**Upper = x̄ + t₍₁₋α⁄₂, n−1₎ × (s/√n)**

For a 95% Confidence Interval:

**α = 0.05**

**t-critical = t₍0.975, n−1₎**

Therefore:

**95% CI = x̄ ± t₍0.975, n−1₎ × (s/√n)**

A confidence interval represents uncertainty in our estimate of the population mean.

---

# ENGINEERING IMPLEMENTATION GOAL

For every concept, we will:

1. Understand the mathematical problem.
2. Understand the formula.
3. Implement it manually using NumPy.
4. Test edge cases.
5. Validate against NumPy/SciPy.
6. Interpret the statistical result.
7. Apply it to real financial data.
8. Later convert useful statistical insights into ML features.

Our complete pipeline is:

**Financial Data**

↓

**Mathematics**

↓

**NumPy Implementation**

↓

**Statistical Validation**

↓

**Interpretation**

↓

**Feature Engineering**

↓

**Machine Learning**

The goal is not simply to call statistical functions.

The goal is to understand:

**What the mathematics means → 
How it is implemented → 
What the result tells us → 
How it helps Machine Learning**
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats


# =====================================================================
# 1. VECTORIZED CASH-FLOW & VOLATILITY CALCULATIONS
# =====================================================================

def calculate_cash_flow_trajectory(
    income_vector: np.ndarray,
    expense_vector: np.ndarray,
    initial_balance: float = 0.0,
) -> np.ndarray:
    """
    Computes the continuous bank balance trajectory:
    B_t = B_0 + cumsum(Income - Expense)
    """
    net_daily_flow = income_vector-expense_vector
    balance_trajectory = initial_balance + np.cumsum(net_daily_flow)
    return balance_trajectory

def calculate_volatility_metrics(amounts: np.ndarray) -> Dict[str,float] :
    """
    Computes scale-independent volatility and dispersion metrics:
    Mean, Std Dev, Variance, and Coefficient of Variation (CV = std / mean).
    """

    if len(amounts) < 2:
        return {"mean":0.0, "std":0.0, "variance":0.0,  "cv": 0.0}

    mean_val = float(np.mean(amounts))
    std_val = float(np.std(amounts,ddof=1)) # ddof = 1 for sample unbiased estimator
    """
    ddof stands for Delta Degrees of Freedom.

In NumPy:
If you write np.std(x) without arguments, NumPy assumes ddof=0 (dividing by $N$). This is the Population Standard Deviation ($\sigma = \sqrt{\frac{1}{N}\sum (x_i - \mu)^2}$).
But in real life and ML, our data is always a sample of a user's behavior, not their entire lifetime of transactions.
Because the sample mean $\bar{x}$ is calculated from the same sample, dividing by $N$ systematically underestimates the true spread (this is called statistical bias).
To get an unbiased estimator (Bessel’s Correction), we divide by $N - 1$.
In NumPy: divisor = N - ddof. Therefore, setting ddof=1 instructs NumPy to divide by $N - 1$.
"""
    var_val = float(np.var(amounts,ddof = 1))
    cv_val = float(std_val/mean_val) if mean_val > 0 else 0.0

    return {
        "mean": round(mean_val, 2),
        "std": round(std_val, 2),
        "variance": round(var_val, 2),
        "cv": round(cv_val, 4),
    }


# =====================================================================
# 2. CATEGORY CORRELATION & COVARIANCE
# =====================================================================

def compute_category_correlation_matrix(df_monthly_pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the Pearson correlation matrix between monthly category expenses:
    r_xy = Cov(X, Y) / (sigma_X * sigma_Y)
    """
    return df_monthly_pivot.corr(method = "pearson").round(3)
 

# =====================================================================
# 3. OUTLIER DETECTION ENGINE (Z-Score, Modified Z-Score, IQR)
# =====================================================================

def detect_outliers_zscore(amounts: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Standard Z-Score Outlier Filter:
    z = (x - mean) / std. Returns boolean mask (True where |z| > threshold).
    """

    if len(amounts) < 3:
        return np.zeros(len(amounts), dtype = bool)

    mean = np.mean(amounts)
    std = np.std(amounts, ddof = 1)
    if std == 0:
        return np.zeros(len(amounts), dtype = bool)
    
    z_score = np.abs((amounts - mean)/ std)
    return z_score > threshold


def detect_outliers_modified_zscore(amounts: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """
    Robust Modified Z-Score using Median Absolute Deviation (MAD):
    MAD = median(|x_i - median(x)|)
    M_i = 0.6745 * (x_i - median(x)) / MAD
    Resistant to extreme outlier distortion.
    """
    
    if len(amounts) < 3:
        return np.zeros(len(amounts), dtype=bool)

    median = np.median(amounts)
    abs_deviations = np.abs(amounts - median)
    mad = np.median(abs_deviations)

    if mad == 0:
        return np.zeros(len(amounts), dtype=bool)
    modified_z_scores = 0.6745 * abs_deviations / mad
    return modified_z_scores > threshold

'''
The MAD (Median Absolute Deviation) & Modified Z-Score:
This is one of the most important statistical concepts in financial anomaly detection!

The Problem with standard Z-Score:
Imagine a user's normal daily grocery spends are: [1000, 1100, 1050, 1020, 1080] (Mean $\approx$ 1050, Std $\approx$ 40). Now imagine on day 6 they suffer an extreme hospital emergency shock of ₹100,000.

If we calculate standard Z-score:

The ₹100,000 pulls the mean $\bar{x}$ up from ₹1,050 to ₹17,540!
The ₹100,000 pulls the standard deviation $s$ up from ₹40 to ₹40,400!
Now look at the Z-score of the ₹100,000 shock: $$z = \frac{100000 - 17540}{40400} \approx \mathbf{2.04}$$
Because the outlier inflated the standard deviation, its own Z-score is only 2.04 ($< 3.0$)! The standard Z-score failed to detect the massive anomaly! This is called the masking effect.
The Robust Solution: MAD
Instead of using the mean and standard deviation (which are easily corrupted by extreme numbers), we use the Median:

Median $\tilde{x}$: The middle value. If you inject ₹100,000, the median barely moves (from 1050 to 1065).
Absolute Deviations: $|x_i - \tilde{x}|$.
MAD: $\text{median}(|x_i - \tilde{x}|)$.
Why multiply by $0.6745$?:
For a standard normal distribution, the 75th percentile is at $z \approx 0.6745$.
Thus, for normally distributed data, $\text{MAD} \approx 0.6745 \cdot \sigma$, which means $\sigma \approx \frac{\text{MAD}}{0.6745}$.
Dividing deviation by this estimated sigma gives: $$M_i = \frac{0.6745 \cdot |x_i - \tilde{x}|}{\text{MAD}}$$
If $|M_i| > 3.5$, it is flagged as an outlier with mathematical robustness.
'''

def detect_outliers_iqr(amounts : np.ndarray, multiplier: float = 1.5) -> Tuple[np.ndarray, float, float]:
    """
    Tukey's IQR Outlier Detection:
    IQR = Q3 - Q1
    Bounds = [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    Returns: (boolean_mask, lower_bound, upper_bound)
    """

    if len(amounts) < 4:
        return np.zeros(len(amounts),dtype = bool),0.0,0.0

    q1 = float(np.percentile(amounts,25))
    q3 = float(np.percentile(amounts,75))
    iqr = q3 - q1
    lower = q1 - multiplier*iqr
    upper = q3 + multiplier*iqr
    outlier_mask = (amounts < lower) | (amounts > upper)
    return outlier_mask, lower, upper

# =====================================================================
# 4. CONFIDENCE INTERVAL ESTIMATION
# =====================================================================

def calculate_expense_confidence_interval(
    monthly_expenses: np.ndarray,
    confidence_level: float = 0.95,
)-> Dict[str, float]:
    """
    Calculates the Student-t Confidence Interval for expected monthly expenses:
    CI = mean +/- t_(alpha/2, n-1) * (s / sqrt(n))
    """
    
    n = len(monthly_expenses)
    if( n < 2):
        val = float(monthly_expenses[0]) if n == 1 else 0.0
        return{
            "mean" : val,
            "ci_lower": val,
            "ci_upper": val,
            "margin_error": 0.0,
        }

    mean_val = float(np.mean(monthly_expenses))
    std_err = stats.sem(monthly_expenses)  # Standard Error = s / sqrt(n)

    # Degree of freedom =  n-1
    t_critical = float(stats.t.ppf((1+confidence_level)/2.0,df = n-1))
    margin_of_error = t_critical * std_err
    ci_lower = mean_val - margin_of_error
    ci_upper = mean_val + margin_of_error

    return{
        "mean":round(mean_val,2),
        "ci_lower": round(ci_lower,2),
        "ci_upper": round(ci_upper,2),
        "margin_of_error": round(margin_of_error,2),
        "confidence_level": confidence_level,
    }
    

# =====================================================================
# SANITY TEST BLOCK
# =====================================================================
if __name__ == "__main__":
    print("--- 1. Testing Cash Flow & Volatility ---")
    incomes = np.array([80000, 0, 0, 5000, 0, 0, 0])
    expenses = np.array([25000, 1200, 800, 3500, 950, 4200, 1500])
    balance = calculate_cash_flow_trajectory(incomes, expenses, initial_balance =50000)
    print(f"Net Ending Balance: ₹{balance[-1]:,.2f}")
    volatility = calculate_volatility_metrics(expenses)
    print(f"Expense Volatility (CV): {volatility['cv']:.4f} (Mean: ₹{volatility['mean']}, Std: ₹{volatility['std']})")
    print("\n--- 2. Testing Outlier Detection on Synthetic Spikes ---")
    sample_spends = np.array([1200, 1500, 1100, 1400, 1350, 1250, 1600, 1300, 85000, 1400])  # 85k shock
    z_outliers = detect_outliers_zscore(sample_spends)
    mad_outliers = detect_outliers_modified_zscore(sample_spends)
    iqr_outliers, lb, ub = detect_outliers_iqr(sample_spends)
    
    print(f"Sample Spends: {sample_spends}")
    print(f"Z-Score Outlier Index: {np.where(z_outliers)[0]}")
    print(f"MAD Outlier Index:     {np.where(mad_outliers)[0]}")
    print(f"IQR Outlier Index:     {np.where(iqr_outliers)[0]} (Bounds: ₹{lb} to ₹{ub})")
    print("\n--- 3. Testing 95% Confidence Interval ---")
    monthly_spends = np.array([48000, 52000, 49500, 56000, 51000, 47500, 53000, 50500])
    ci = calculate_expense_confidence_interval(monthly_spends, confidence_level=0.95)
    print(f"Expected Next Month Spend: ₹{ci['mean']:,.2f} [95% CI: ₹{ci['ci_lower']:,.2f} - ₹{ci['ci_upper']:,.2f}]")

        