# Day 5: Portfolio Optimization & Risk Management

#### Overview
On Day 5, we will explore portfolio optimization and risk management, covering key concepts, mathematical formulations, and implementations in both Python and C++. The focus will be on Modern Portfolio Theory (MPT), the Capital Asset Pricing Model (CAPM), Value at Risk (VaR), and different optimization techniques.

---

### **1. Portfolio Optimization**

#### **1.1 Modern Portfolio Theory (MPT)**
MPT, developed by Harry Markowitz, aims to construct an optimal portfolio that maximizes expected return for a given level of risk.

- **Expected Return**:
\[
E(R_p) = \sum_{i=1}^{n} w_i E(R_i)
\]
Where:
- \(E(R_p)\): Expected return of the portfolio
- \(w_i\): Weight of asset \(i\) in the portfolio
- \(E(R_i)\): Expected return of asset \(i\)

- **Portfolio Variance**:
\[
\sigma^2_p = \sum_{i=1}^{n} w_i^2 \sigma^2_i + \sum_{i \neq j} w_i w_j \sigma_{ij}
\]
Where:
- \(\sigma^2_p\): Variance of the portfolio
- \(\sigma^2_i\): Variance of asset \(i\)
- \(\sigma_{ij}\): Covariance between assets \(i\) and \(j\)

#### **1.2 Portfolio Optimization Problem**
The objective is to maximize the Sharpe Ratio, defined as:
\[
\text{Sharpe Ratio} = \frac{E(R_p) - R_f}{\sigma_p}
\]
Where:
- \(R_f\): Risk-free rate

**Python Code Example** (Using `numpy` and `scipy`):
```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Sample data
returns = pd.DataFrame({
    'Asset1': np.random.normal(0.01, 0.02, 1000),
    'Asset2': np.random.normal(0.015, 0.025, 1000),
    'Asset3': np.random.normal(0.012, 0.022, 1000)
})

# Calculate expected returns and covariance matrix
expected_returns = returns.mean()
cov_matrix = returns.cov()

# Portfolio optimization function
def portfolio_performance(weights):
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_std_dev = np.sqrt(portfolio_variance)
    return portfolio_return, portfolio_std_dev

# Objective function to minimize (negative Sharpe Ratio)
def negative_sharpe(weights):
    portfolio_return, portfolio_std_dev = portfolio_performance(weights)
    return - (portfolio_return - 0.01) / portfolio_std_dev  # Assuming risk-free rate = 1%

# Constraints and bounds
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
bounds = tuple((0, 1) for _ in range(len(expected_returns)))

# Initial guess
init_guess = [1./len(expected_returns)] * len(expected_returns)

# Optimize
optimal = minimize(negative_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = optimal.x
print(f'Optimal Weights: {optimal_weights}')
```

**C++ Code Example** (Using `Eigen` for matrix operations):
```cpp
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace Eigen;

double portfolio_return(const VectorXd& weights, const VectorXd& expected_returns) {
    return weights.dot(expected_returns);
}

double portfolio_variance(const VectorXd& weights, const MatrixXd& cov_matrix) {
    return (weights.transpose() * cov_matrix * weights)(0, 0);
}

double negative_sharpe(const VectorXd& weights, const VectorXd& expected_returns, const MatrixXd& cov_matrix, double risk_free_rate) {
    double port_return = portfolio_return(weights, expected_returns);
    double port_variance = portfolio_variance(weights, cov_matrix);
    double port_std_dev = std::sqrt(port_variance);
    return - (port_return - risk_free_rate) / port_std_dev;
}

int main() {
    VectorXd expected_returns(3);
    expected_returns << 0.01, 0.015, 0.012;

    MatrixXd cov_matrix(3, 3);
    cov_matrix << 0.0004, 0.0001, 0.0002,
                  0.0001, 0.000625, 0.00015,
                  0.0002, 0.00015, 0.000484;

    // Optimize weights (using a simple grid search for demonstration)
    VectorXd best_weights(3);
    double best_sharpe = -INFINITY;
    
    for (double w1 = 0; w1 <= 1; w1 += 0.1) {
        for (double w2 = 0; w2 <= (1 - w1); w2 += 0.1) {
            double w3 = 1 - w1 - w2;
            VectorXd weights(3);
            weights << w1, w2, w3;
            double current_sharpe = -negative_sharpe(weights, expected_returns, cov_matrix, 0.01);
            if (current_sharpe > best_sharpe) {
                best_sharpe = current_sharpe;
                best_weights = weights;
            }
        }
    }

    std::cout << "Optimal Weights: " << best_weights.transpose() << std::endl;
    return 0;
}
```

---

### **2. Risk Management**

#### **2.1 Value at Risk (VaR)**
VaR quantifies the potential loss in value of a portfolio over a defined period for a given confidence interval.

- **Parametric VaR**:
\[
\text{VaR}_{\alpha} = \mu - z_{\alpha} \sigma
\]
Where:
- \(\mu\): Mean return of the portfolio
- \(\sigma\): Standard deviation of returns
- \(z_{\alpha}\): z-score corresponding to the confidence level \(\alpha\)

**Python Code Example**:
```python
def calculate_var(returns, confidence_level=0.95):
    mean = returns.mean()
    std_dev = returns.std()
    z_score = norm.ppf(confidence_level)
    var = mean - z_score * std_dev
    return var

var = calculate_var(returns.mean())
print(f'Value at Risk (VaR): {var}')
```

**C++ Code Example**:
```cpp
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <boost/math/distributions/normal.hpp>

double calculate_var(const VectorXd& returns, double confidence_level) {
    double mean = returns.mean();
    double std_dev = std::sqrt((returns.array() - mean).square().mean());
    double z_score = boost::math::quantile(boost::math::normal(), confidence_level);
    return mean - z_score * std_dev;
}

int main() {
    VectorXd returns(5);
    returns << 0.01, 0.02, -0.01, 0.015, -0.005;

    double var = calculate_var(returns, 0.95);
    std::cout << "Value at Risk (VaR): " << var << std::endl;
    return 0;
}
```

#### **2.2 Other Risk Measures**
- **Conditional Value at Risk (CVaR)**: Measures the average loss exceeding VaR.
- **Maximum Drawdown**: Measures the maximum observed loss from a peak to a trough.

---

### **3. Questions & Answers Section**

#### **Q1: Explain the concept of Sharpe Ratio and its significance.**
**A1:** The Sharpe Ratio measures the risk-adjusted return of an investment, defined as the difference between the return of the investment and the risk-free rate divided by the investment's standard deviation. A higher Sharpe Ratio indicates better risk-adjusted performance.

#### **Q2: How is the portfolio variance calculated?**
**A2:** Portfolio variance is calculated as the weighted sum of the variances of individual assets and the covariances between them, using the formula:
\[
\sigma^2_p = \sum_{i=1}^{n} w_i^2 \sigma^2_i + \sum_{i \neq j} w_i w_j \sigma_{ij}
\]

#### **Q3: Write a Python function to calculate VaR for a given return series.**
**A3:**
```python
def calculate_var(returns, confidence_level=0.95):
    mean = returns.mean()
    std_dev = returns.std()
    z_score = norm.ppf(confidence_level)
    var = mean - z_score * std_dev
    return var

var = calculate_var(returns.mean())
print(f'Value at Risk (VaR): {var}')
```

#### **Q4: Discuss the importance of risk management in trading.**
**A4:** Risk management is crucial in trading as it helps protect against potential losses. It allows traders to set limits on losses, diversify portfolios, and maintain a balanced risk-reward ratio, which is essential for long-term success.

---

### Conclusion
By the end of Day 5, you should have a solid understanding of portfolio optimization techniques, risk management measures, and

 how to implement them in Python and C++. These tools are essential for any quantitative trading strategy aimed at maximizing returns while managing risk effectively.
