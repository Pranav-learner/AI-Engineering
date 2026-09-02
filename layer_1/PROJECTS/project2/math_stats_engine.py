"""
Machine learning models are only as good as the mathematical signals fed into them. In high-stakes fraud detection, we must combine probabilistic reasoning, distributional anomaly detection, and physical geometry:

1. Bayes' Theorem & Likelihood Ratios: How prior fraud probabilities update dynamically given new evidence (e.g., sudden midnight login).
2. Benford's Law ($\chi^2$ Goodness-of-Fit): Leading digit distribution $P(d) = \log_{10}(1 + 1/d)$ — natural transaction amounts strictly obey Benford's law, while synthetic/bot/tampered fraud amounts heavily violate it.
3. Robust Mahalanobis Distance ($D_M$): Detects multi-dimensional correlated anomalies (e.g. high amount + high velocity) accounting for feature covariance, where standard Euclidean distance fails.
4. Modified Z-Score via Median Absolute Deviation (MAD): Immune to extreme outlier contamination, unlike standard mean/std.
5. Haversine Geodesic Velocity: Calculates distance and velocity ($\text{km/h}$) between consecutive transactions to detect Impossible Travel (e.g., Mumbai at 12:00 $\to$ London at 12:45).
6. Shannon Entropy: Quantifies behavioral randomness across merchant categories and channels.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.stats as stats


# =====================================================================
# 1. BAYESIAN INFERENCE & ODDS RATIO ENGINE
# =====================================================================
def calculate_posterior_fraud_probability(
    prior_fraud_prob: float,
    likelihood_given_fraud: float,
    likelihood_given_legit: float,
) -> float:
    """
    Computes Posterior Probability P(Fraud | Evidence) via Bayes' Theorem:
    
                        P(Evidence | Fraud) * P(Fraud)
      P(Fraud|Evidence) = ─────────────────────────────────────────────────────────────
                        P(Evidence | Fraud)*P(Fraud) + P(Evidence | Legit)*P(Legit)
    Why we use it:
    A single suspicious signal (e.g., new device) does not mean 100% fraud.
    Bayes' theorem formally balances the rare prior P(Fraud)=0.008 with the observed evidence.
    """
    prior_legit_prob = 1.0 - prior_fraud_prob
    
    # Numerator: Joint probability of observing evidence AND fraud occurring
    numerator = likelihood_given_fraud * prior_fraud_prob
    
    # Denominator: Total evidence probability (marginal likelihood)
    denominator = numerator + (likelihood_given_legit * prior_legit_prob)
    
    if denominator == 0.0:
        return 0.0
        
    return float(numerator / denominator)


def bayes_odds_update(prior_prob: float, likelihood_ratio: float) -> Tuple[float, float]:
    """
    Converts prior probability to prior odds, applies the Likelihood Ratio (Bayes Factor),
    and converts posterior odds back to posterior probability:
    
      Prior Odds = P / (1 - P)
      Posterior Odds = Prior Odds * Likelihood Ratio
      Posterior P = Posterior Odds / (1 + Posterior Odds)
    """
    if prior_prob >= 1.0:
        return float("inf"), 1.0
    if prior_prob <= 0.0:
        return 0.0, 0.0
        
    prior_odds = prior_prob / (1.0 - prior_prob)
    posterior_odds = prior_odds * likelihood_ratio
    posterior_prob = posterior_odds / (1.0 + posterior_odds)
    
    return float(posterior_odds), float(posterior_prob)


# =====================================================================
# 2. BENFORD'S LAW LEADING-DIGIT ANOMALY ENGINE
# =====================================================================
def benford_expected_distribution() -> np.ndarray:
    """
    Returns theoretical Benford's Law distribution for leading digits d in {1, 2, ..., 9}:
      P(d) = log10(1 + 1/d)
      
    Digit probabilities:
      1: 30.1%, 2: 17.6%, 3: 12.5%, 4: 9.7%, 5: 7.9%, 6: 6.7%, 7: 5.8%, 8: 5.1%, 9: 4.6%
    """
    digits = np.arange(1, 10)
    return np.log10(1.0 + 1.0 / digits)
def calculate_benford_conformance(amounts: Union[List[float], np.ndarray]) -> Dict[str, float]:
    """
    Performs Chi-Square Goodness-of-Fit test on transaction amounts against Benford's Law.
    
    Why this detects fraud:
    Human fraudsters and automated attack scripts tend to generate amounts with uniform
    or rounded leading digits (e.g., ₹4999, ₹5000, ₹999), strongly violating Benford's
    logarithmic distribution.
    
    Returns:
      - chi2_stat: Chi-Square test statistic (higher = greater deviation from natural distribution)
      - p_value: Probability that the observed amounts conform to natural logarithmic scaling
      - is_anomalous: Boolean flag (p_value < 0.01 indicates tampered distribution)
    """
    arr = np.asarray(amounts, dtype=float)
    arr = arr[arr > 0] # Filter non-positive values
    
    if len(arr) < 30:
        # Not enough sample size to reliably test Chi-Square distribution (requires N >= 30)
        return {"chi2_stat": 0.0, "p_value": 1.0, "is_anomalous": 0.0}
        
    # Extract leading first digit for each transaction amount
    # Example: 4850.0 -> str '4850.0' -> '4' -> int 4
    first_digits = np.array([int(str(f"{x:.6e}")[0]) for x in arr])
    
    # Compute observed counts for digits 1 through 9
    observed_counts = np.array([np.sum(first_digits == d) for d in range(1, 10)])
    
    total_n = len(first_digits)
    expected_counts = benford_expected_distribution() * total_n
    
    # Chi-Square statistic: sum((Observed - Expected)^2 / Expected)
    # df = 9 - 1 = 8 degrees of freedom
    chi2_stat, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)
    
    return {
        "chi2_stat": float(chi2_stat),
        "p_value": float(p_value),
        "is_anomalous": 1.0 if p_value < 0.01 else 0.0,
    }


# =====================================================================
# 3. MULTIVARIATE ANOMALY DETECTION (ROBUST MAHALANOBIS DISTANCE)
# =====================================================================
def calculate_mahalanobis_distance(
    X: np.ndarray,
    reference_mean: Optional[np.ndarray] = None,
    reference_cov: Optional[np.ndarray] = None,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Calculates the Mahalanobis Distance for multivariate feature vectors:
    
      D_M(x) = sqrt( (x - mu)^T * Sigma^{-1} * (x - mu) )
      
    Why Euclidean distance is wrong for fraud:
    Transaction amount (scale: 10,000) and Velocity (scale: 5) have completely different variances
    and are often correlated. Euclidean distance treats all axes equally and ignores correlation.
    Mahalanobis distance standardizes axes by covariance Sigma, projecting data onto principal ellipsoids.
    
    Parameters:
      - X: Matrix of shape (N, D) where N is number of samples, D is feature dimensions
      - epsilon: Ridge shrinkage parameter added to diagonal (Sigma + eps*I) to prevent singular matrix inversion
    """
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)
        
    N, D = X_arr.shape
    
    # 1. Compute or use reference mean
    if reference_mean is None:
        mu = np.mean(X_arr, axis=0)
    else:
        mu = np.asarray(reference_mean, dtype=float)
        
    # 2. Compute or use reference covariance
    if reference_cov is None:
        cov = np.cov(X_arr, rowvar=False)
        if D == 1:
            cov = np.array([[cov]])
    else:
        cov = np.asarray(reference_cov, dtype=float)
        
    # 3. Regularize covariance matrix to guarantee invertible positive-definiteness
    cov_reg = cov + np.eye(D) * epsilon
    inv_cov = np.linalg.pinv(cov_reg)
    
    # 4. Vectorized computation of Mahalanobis distance
    diff = X_arr - mu # Shape: (N, D)
    
    # (diff @ inv_cov) * diff -> elementwise row product summed across columns
    left_term = np.dot(diff, inv_cov) # (N, D)
    mahal_sq = np.sum(left_term * diff, axis=1) # (N,)
    
    # Numerical guard: clip negative values arising from floating point errors before sqrt
    mahal_sq = np.clip(mahal_sq, a_min=0.0, a_max=None)
    return np.sqrt(mahal_sq)
# =====================================================================
# 4. ROBUST UNIVARIATE OUTLIER DETECTION (MODIFIED Z-SCORE VIA MAD)
# =====================================================================
def detect_outliers_modified_zscore(
    values: Union[List[float], np.ndarray],
    threshold: float = 3.5,
) -> np.ndarray:
    """
    Computes Boris Iglewicz and David Hoaglin's Modified Z-Score:
    
      M_i = 0.6745 * (x_i - Median) / MAD
      
    where MAD = Median( |x_i - Median| ).
    The constant 0.6745 is the expected MAD for a standard normal distribution.
    
    Why MAD outperforms standard Z-score:
    Standard mean and standard deviation are heavily corrupted by extreme fraud amounts
    (e.g., one ₹500,000 fraud corrupts the mean of 50 normal ₹500 transactions).
    Median and MAD are mathematically robust to up to 50% data contamination.
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return np.array([], dtype=bool)
        
    median = np.median(arr)
    abs_deviation = np.abs(arr - median)
    mad = np.median(abs_deviation)
    
    if mad == 0.0:
        # If >50% values are identical, use mean absolute deviation to avoid div-by-zero
        mean_ad = np.mean(abs_deviation)
        if mean_ad == 0.0:
            return np.zeros(len(arr), dtype=bool)
        modified_z = 0.6745 * (arr - median) / mean_ad
    else:
        modified_z = 0.6745 * (arr - median) / mad
        
    return np.abs(modified_z) > threshold
# =====================================================================
# 5. GEODESIC DYNAMICS (HAVERSINE DISTANCE & IMPOSSIBLE VELOCITY)
# =====================================================================
def haversine_distance_km(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """
    Calculates Great-Circle Distance (in kilometers) between two GPS points on Earth
    using the Haversine formula:
    
      a = sin^2(dlat/2) + cos(lat1)*cos(lat2)*sin^2(dlon/2)
      c = 2 * arcsin( sqrt(a) )
      d = R * c   (Earth radius R = 6371.0 km)
    """
    # Convert degrees to radians
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    # Haversine formula
    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * (np.sin(delta_lambda / 2.0) ** 2)
    )
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    
    earth_radius_km = 6371.0
    return float(earth_radius_km * c)
def calculate_travel_velocity_kmh(
    lat1: float, lon1: float, timestamp1_sec: float,
    lat2: float, lon2: float, timestamp2_sec: float,
) -> float:
    """
    Calculates implied physical travel speed (in km/h) between two consecutive transactions.
    
    Why this catches fraud:
    If Card X is swiped in Bangalore at 14:00 and then swiped online/POS in London at 14:45,
    distance = ~7,500 km in 0.75 hours -> velocity = 10,000 km/h.
    Since commercial passenger aircraft cruise at ~900 km/h, any velocity > 900 km/h is
    physically impossible and represents cloned card or credential sharing.
    """
    delta_time_hours = (timestamp2_sec - timestamp1_sec) / 3600.0
    
    if delta_time_hours <= 0.0:
        # Simultaneous transactions at different locations
        dist = haversine_distance_km(lat1, lon1, lat2, lon2)
        return 99999.0 if dist > 5.0 else 0.0
        
    distance_km = haversine_distance_km(lat1, lon1, lat2, lon2)
    return float(distance_km / delta_time_hours)
# =====================================================================
# 6. SHANNON ENTROPY (BEHAVIORAL RANDOMNESS)
# =====================================================================
def calculate_shannon_entropy(categories: List[str]) -> float:
    """
    Computes Shannon Information Entropy H(X) for categorical sequence:
    
      H(X) = - sum( p(x_i) * log2( p(x_i) ) )
      
    Why we use it:
    A legitimate user typically shops at predictable categories (low entropy).
    When an account is compromised or bot-swept, transactions scatter erratically
    across unusual channels and merchants, causing an entropy spike.
    """
    if len(categories) == 0:
        return 0.0
        
    _, counts = np.unique(categories, return_counts=True)
    probabilities = counts / len(categories)
    
    # Filter out zero probabilities to avoid log2(0)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))
