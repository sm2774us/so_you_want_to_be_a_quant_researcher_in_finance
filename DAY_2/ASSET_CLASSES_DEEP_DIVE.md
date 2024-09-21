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

### **7. Derivatives (Futures, Swaps)**
   - **Futures Contracts** are standardized contracts to buy/sell an asset at a future date.
   - **Swaps** involve the exchange of cash flows (e.g., fixed vs floating interest rate).

   #### **Mathematical Models**:
   - **Pricing Futures**: Typically done using the **Cost of Carry Model**.
   - **Swap Pricing**: Involves discounting the fixed leg and the floating leg to present value.

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

---

### **8. Crypto**
   - **Crypto Assets** are digital and decentralized. Analysis of crypto involves blockchain data and price patterns.
   - **Mathematical Models**: Volatility in crypto is often modeled using GARCH and stochastic models similar to FX.

   #### **Statistical Tools**:
   - **Moving Average**: Used to smooth out price data.
   - **Volatility Metrics**: GARCH models to estimate volatility.
   - **Correlation Analysis**: To assess how crypto moves relative to other assets (e.g., stocks, commodities).

---

These are the key mathematical and statistical models for various asset classes. Each asset requires specific tools and techniques for pricing, risk management, and trading strategy development.

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>
