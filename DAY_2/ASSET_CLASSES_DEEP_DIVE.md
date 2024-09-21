# __Asset Classes - Deep Dive__

To dive deeper into the section on **Asset Classes**, let's provide a detailed explanation for each asset category, focusing on the mathematics, statistics, and important models or methodologies used in analyzing them. This will include concepts like **Options (Greeks, American vs European options, Black-Scholes method)** and **Interest Rates (curve construction methodologies)**, among others.

<div align="right"><a href="./README.md" target="_blacnk"><img src="https://img.shields.io/badge/Back To Day 2 Overview-blue?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

---

### **Day-2 Plan Overview** <a name="top"></a>

1. [**1. Equities**](#1-equities)
2. [**2. Options**](#2-options)
3. [**3. Fixed Income (Bonds and Treasuries)**](#3-fixed-income-bonds-and-treasuries)
4. [**4. Interest Rates**](#4-interest-rates)
5. [**5. Commodities**](#5-commodities)
6. [**6. Foreign Exchange (FX)**](#6-foreign-exchange-fx)
7. [**7. Derivatives (Futures, Swaps)**](#7-derivatives-futures-swaps)
8. [**8. Crypto**](#8-crypto)

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

---

### **1. Equities**
   - **Equities** represent ownership in a company. The price of equities fluctuates based on the company's performance and external market conditions. Analysis often involves:
     - **Price Models**: Geometric Brownian Motion (GBM), CAPM (Capital Asset Pricing Model).
     - **Key Metrics**: Sharpe Ratio, Beta (correlation to market returns), Alpha.
     - **Statistical Tools**:
       - **Expected Return**: $$E(R) = \sum p_i R_i$$
       - **Volatility (Standard Deviation)**: $$\sigma = \sqrt{\sum p_i (R_i - E(R))^2}$$
       - **Correlation**: $$\rho_{X,Y} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

   - **Example**: Calculate the **Beta** of a stock relative to the market.
     - $$\beta = \frac{\text{Cov}(R_{\text{stock}}, R_{\text{market}})}{\sigma_{\text{market}}^2}$$

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

---

### **2. Options**
   **Options** give the right (but not the obligation) to buy/sell an asset at a set price in the future. These are analyzed using sophisticated mathematical models.

   #### **Types of Options**:
   - **American Options**: Can be exercised at any time before expiry.
   - **European Options**: Can only be exercised at expiry.

   #### **Option Greeks**:
   These are derivatives of the option price with respect to various factors:
   - **Delta (Δ)**: Sensitivity of the option price to changes in the underlying asset price.
     - $$\Delta = \frac{\partial V}{\partial S}$$
   - **Gamma (Γ)**: Sensitivity of Delta to changes in the underlying asset price.
     - $$\Gamma = \frac{\partial^2 V}{\partial S^2}$$
   - **Theta (Θ)**: Sensitivity to time decay.
     - $$\Theta = \frac{\partial V}{\partial t}$$
   - **Vega (ν)**: Sensitivity to volatility of the underlying asset.
     - $$\nu = \frac{\partial V}{\partial \sigma}$$
   - **Rho (ρ)**: Sensitivity to interest rate changes.
     - $$\rho = \frac{\partial V}{\partial r}$$

   #### **Black-Scholes Model** (for European options):
A famous model for pricing European options under certain assumptions (constant volatility, interest rate, no dividends):
- **Formula** for a call option price $$C$$:
```math
C = S_0 \Phi(d_1) - K e^{-rT} \Phi(d_2)
```
where,
```math
d_1 = \frac{\ln\left(\frac{S_0}{K}\right) + (r + \frac{\sigma^2}{2})T}{\sigma \sqrt{T}}, \quad d_2 = d_1 - \sigma \sqrt{T}
```
- $$S_0$$: Current stock price
- $$K$$: Strike price
- $$T$$: Time to maturity
- $$r$$: Risk-free interest rate
- $$\sigma$$: Volatility
- $$\Phi(\cdot)$$: Cumulative distribution function of the standard normal distribution.

   #### **American Option Pricing**:
   - American options can’t use the Black-Scholes model directly since early exercise is allowed.
   - **Binomial Trees**: Used to model multiple paths the price of an asset could take, accounting for the possibility of early exercise.

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

---

### **3. Fixed Income (Bonds and Treasuries)**
   - **Fixed Income Securities** are debt instruments that pay a fixed return over time. Pricing and yield calculations are crucial here.
   
   #### **Bond Pricing**:
- Bond prices are calculated by discounting future cash flows (coupons and face value) to the present.
- **Price of Bond**:
```math
P = \sum \frac{C}{(1 + r)^t} + \frac{F}{(1 + r)^T}
```

where $$C$$ is the coupon payment, $$F$$ is the face value, $$r$$ is the discount rate, and $$t$$ is time to payment.

   #### **Duration and Convexity**:
- **Duration** measures sensitivity to interest rate changes:
```math
D = \frac{1}{P} \sum \frac{t \cdot C}{(1 + r)^t}
```
- **Convexity** accounts for the curvature in price/yield relationship:
```math
C = \frac{1}{P} \sum \frac{t(t+1) \cdot C}{(1 + r)^{t+2}}
```

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

---

### **4. Interest Rates**
   Interest rates are crucial for pricing many financial instruments. **Yield Curve Construction** is the backbone of the interest rate market.

   #### **Yield Curve**:
   The yield curve shows the relationship between bond yields and their maturity dates.
   - **Bootstrapping Method**: Used to construct the yield curve from bond prices, by solving for zero-coupon yields.
   - **Forward Rates**: Future interest rates implied by the yield curve.

   #### **Mathematical Models**:
- **Nelson-Siegel Model**: A parametric model to estimate the yield curve:
```math
     y(t) = \beta_0 + \beta_1 \frac{1 - e^{-\frac{t}{\tau}}}{\frac{t}{\tau}} + \beta_2 \left( \frac{1 - e^{-\frac{t}{\tau}}}{\frac{t}{\tau}} - e^{-\frac{t}{\tau}} \right)
```
- **Cox-Ingersoll-Ross (CIR) Model**: A mean-reverting stochastic process for modeling interest rate dynamics:
```math
dr_t = \kappa (\theta - r_t) dt + \sigma \sqrt{r_t} dW_t
```
where $$r_t$$ is the short rate, $$\kappa$$ is the speed of mean reversion, $$\theta$$ is the long-term mean, and $$W_t$$ is a Wiener process.

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

---

### **5. Commodities**
   Commodities are physical assets like oil, gold, and agricultural products. They have unique features compared to financial assets due to storage costs, transportation, and physical characteristics.

   #### **Mathematical Models**:
   - **Contango**: When the futures price is higher than the spot price (indicating storage costs).
   - **Backwardation**: When the futures price is lower than the spot price.

   #### **Pricing Models**:
- **Futures Pricing Formula**:
```math
F_t = S_t e^{(r + c)T}
```
where $$S_t$$ is the spot price, $$r$$ is the risk-free rate, $$c$$ is the storage cost, and $$T$$ is the time to maturity.

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

---

### **6. Foreign Exchange (FX)**
   - **FX Markets** are the largest and most liquid in the world. Pricing involves exchange rates between different currencies.

   #### **Mathematical Models**:
- **Covered Interest Rate Parity**: Ensures there’s no arbitrage between different currencies:
```math
\frac{F_{t}}{S_{t}} = \frac{1 + r_d}{1 + r_f}
```
  where $$F_t$$ is the forward rate, $$S_t$$ is the spot rate, $$r_d$$ and $$r_f$$ are the domestic and foreign interest rates respectively.
- **Uncovered Interest Rate Parity**: Relates interest rate differentials to expected changes in exchange rates.
- **GARCH Models**: Used to model volatility in FX markets.

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

---

### ***7. Derivatives (Futures, Swaps)**
   - **Futures Contracts** are standardized contracts to buy/sell an asset at a future date.
   - **Swaps** involve the exchange of cash flows (e.g., fixed vs floating interest rate).

   #### **Mathematical Models**:
   - **Pricing Futures**: Typically done using the **Cost of Carry Model**.
   - **Swap Pricing**: Involves discounting the fixed leg and the floating leg to present value.

#### 7.1 **Futures Contracts**
A **futures contract** is a standardized agreement to buy or sell a specific quantity of an asset at a predetermined price at a future date.

##### **Mathematical Model: Cost of Carry Model**
The **Cost of Carry** model is used to price futures contracts. It includes costs associated with holding the underlying asset until the delivery date of the futures contract.

The formula for the **Futures Price** is:

```math
F = S e^{(r - d)T}
```

Where:
- $$F$$ = Futures price
- $$S$$ = Spot price (current price of the asset)
- $$r$$ = Risk-free interest rate
- $$d$$ = Dividend yield (if the underlying asset pays dividends)
- $$T$$ = Time to maturity (in years)
- $$e$$ = Euler’s number

This formula assumes that the futures price will converge to the spot price at the time of contract expiration. It also accounts for **carrying costs** like financing costs (interest rates) and dividend payouts.

##### **Example in Python**:
Let’s calculate the futures price of a stock with a spot price of $100, a risk-free rate of 5%, and no dividend yield, for a contract that expires in 1 year.

```python
import math

# Input parameters
spot_price = 100  # current price
risk_free_rate = 0.05  # 5% annual rate
dividend_yield = 0  # no dividends
time_to_maturity = 1  # 1 year

# Cost of Carry Model Formula
futures_price = spot_price * math.exp((risk_free_rate - dividend_yield) * time_to_maturity)

print(f"Futures Price: {futures_price:.2f}")
```

##### **Example in C++**:
```cpp
#include <iostream>
#include <cmath>

int main() {
    double spot_price = 100.0; // current price
    double risk_free_rate = 0.05; // 5% annual rate
    double dividend_yield = 0.0; // no dividends
    double time_to_maturity = 1.0; // 1 year

    // Cost of Carry Model Formula
    double futures_price = spot_price * std::exp((risk_free_rate - dividend_yield) * time_to_maturity);

    std::cout << "Futures Price: " << futures_price << std::endl;
    return 0;
}
```

---

#### 7.2 **Swaps**

A **swap** is a financial contract where two parties exchange cash flows over time. The most common type is an **interest rate swap**, where one party pays a fixed interest rate and the other pays a floating rate (e.g., linked to LIBOR).

##### **Mathematical Model for Swap Pricing: Present Value of Cash Flows**
The price of a swap is based on the present value of cash flows from both legs (fixed and floating). Swaps are typically structured to have zero net present value at inception, meaning the value of both legs is equal.

For a **fixed leg**:

```math
PV_{\text{fixed}} = \sum_{i=1}^{n} \frac{C_{\text{fixed}}}{(1+r_i)^t}
```

Where:
- $$C_{\text{fixed}}$$ = Fixed cash flow payment
- $$r_i$$ = Discount rate (derived from yield curves)
- $$t$$ = Time period for each payment

For a **floating leg**:

```math
PV_{\text{floating}} = \sum_{i=1}^{n} \frac{C_{\text{floating}}}{(1+r_i)^t}
```

Where the floating cash flows change based on a reference rate, such as LIBOR.

##### **Python Code for Swap Pricing (Simplified Fixed vs Floating Swap)**:
```python
import numpy as np

# Input parameters
fixed_rate = 0.03  # 3% fixed rate
floating_rate = 0.025  # LIBOR or other index
notional = 1000000  # $1,000,000 notional
periods = 5  # 5 periods
discount_rates = np.array([0.01, 0.012, 0.014, 0.015, 0.016])  # discount factors

# Calculate present value of the fixed leg
fixed_cash_flows = np.full(periods, fixed_rate * notional)
pv_fixed = np.sum(fixed_cash_flows / (1 + discount_rates)**np.arange(1, periods + 1))

# Calculate present value of the floating leg
floating_cash_flows = np.full(periods, floating_rate * notional)
pv_floating = np.sum(floating_cash_flows / (1 + discount_rates)**np.arange(1, periods + 1))

# Swap value
swap_value = pv_fixed - pv_floating
print(f"Swap Value: {swap_value:.2f}")
```

##### **C++ Code for Swap Pricing (Simplified Fixed vs Floating Swap)**:
```cpp
#include <iostream>
#include <cmath>
#include <vector>

int main() {
    double fixed_rate = 0.03; // 3% fixed rate
    double floating_rate = 0.025; // LIBOR or other index
    double notional = 1000000; // $1,000,000 notional
    int periods = 5;
    std::vector<double> discount_rates = {0.01, 0.012, 0.014, 0.015, 0.016};

    double pv_fixed = 0;
    double pv_floating = 0;

    // Calculate present value of the fixed leg
    for (int i = 0; i < periods; i++) {
        pv_fixed += (fixed_rate * notional) / std::pow(1 + discount_rates[i], i + 1);
    }

    // Calculate present value of the floating leg
    for (int i = 0; i < periods; i++) {
        pv_floating += (floating_rate * notional) / std::pow(1 + discount_rates[i], i + 1);
    }

    double swap_value = pv_fixed - pv_floating;
    std::cout << "Swap Value: " << swap_value << std::endl;

    return 0;
}
```

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

---

### **8. Crypto**
   - **Cryptocurrencies are** **digital assets** built on decentralized blockchain networks.
   - Their price behavior is influenced by factors like market demand, blockchain events (forks), and adoption.
   - Analysis of crypto involves blockchain data and price patterns.
   - **Mathematical Models**: Volatility in crypto is often modeled using GARCH and stochastic models similar to FX.

   #### **Statistical Tools**:
   - **Moving Average**: Used to smooth out price data.
   - **Volatility Metrics**: GARCH models to estimate volatility.
   - **Correlation Analysis**: To assess how crypto moves relative to other assets (e.g., stocks, commodities).



#### **8.1 Volatility Modeling with GARCH**

Since crypto markets are highly volatile, **Generalized Autoregressive Conditional Heteroskedasticity (GARCH)** models are often used to estimate volatility over time. The GARCH model allows volatility to change based on past squared returns and past volatility.

**GARCH(1, 1) model**:

```math
\sigma_t^2 = \alpha_0 + \alpha_1 \epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2
```

Where:
- $$\sigma_t^2$$ = Conditional variance (volatility at time t)
- $$\alpha_0, \alpha_1, \beta_1$$ = Model parameters
- $$\epsilon_{t-1}$$ = Past return shock (error term)

##### **Python Code: GARCH Model Using `arch` library**:
```python
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model

# Fetch historical data for Bitcoin
data = yf.download('BTC-USD', start='2020-01-01', end='2023-01-01')
returns = 100 * data['Close'].pct_change().dropna()

# Fit GARCH(1, 1) model
garch_model = arch_model(returns, vol='Garch', p=1, q=1)
garch_fit = garch_model.fit()

# Forecast future volatility
volatility_forecast = garch_fit.forecast(horizon=5)
print(volatility_forecast.variance[-1:])
```

##### **C++ Code for GARCH Model (Conceptual Example)**
A GARCH model implementation in C++ requires specialized statistical libraries like `Boost`, but the basic flow of a GARCH model could be implemented using template meta-programming. For simplicity, this code does not run GARCH but illustrates a conceptual structure:

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

template<typename T>
T garch_model(const std::vector<T>& returns, T alpha0, T alpha1, T beta1, int horizon) {
    std::vector<T> variance(returns.size(), 0);
    variance[0] = std::pow(returns[0], 2);

    for (size_t i = 1; i < returns.size(); ++i) {
        variance[i] = alpha0 + alpha1 * std::pow(returns[i - 1], 2) + beta1 * variance[i - 1];
    }

    T forecast = variance.back();
    for (int i = 0; i < horizon; ++i) {
        forecast = alpha0 + (alpha1 + beta1) * forecast;
    }

    return forecast;
}

int main() {


    std::vector<double> returns = {0.02, -0.01, 0.015, -0.03}; // example returns
    double alpha0 = 0.01, alpha1 = 0.1, beta1 = 0.85;
    int horizon = 5;

    double forecast = garch_model(returns, alpha0, alpha1, beta1, horizon);
    std::cout << "Forecasted volatility: " << forecast << std::endl;
    return 0;
}
```

#### **8.2 Statistical Tools for Crypto**
- **Moving Average**: Simple Moving Average (SMA) and Exponential Moving Average (EMA) smooth out price fluctuations.
- **Volatility Metrics**: GARCH models are often used to estimate conditional volatility.
- **Correlation Analysis**: Pearson or Spearman correlation is often used to assess relationships between crypto assets and traditional assets like equities or commodities.

##### **Example: Correlation Matrix for Crypto and Stocks in Python**:
```python
assets = ['BTC-USD', 'SPY', 'GLD']  # Bitcoin, S&P500, Gold
prices = yf.download(assets, start='2020-01-01')['Close'].pct_change().dropna()
correlation_matrix = prices.corr()
print(correlation_matrix)
```

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

---

These are the key mathematical and statistical models for various asset classes. Each asset requires specific tools and techniques for pricing, risk management, and trading strategy development.

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>
