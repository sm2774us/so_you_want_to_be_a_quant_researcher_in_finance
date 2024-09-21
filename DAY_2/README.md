# **Day 2: Trading Indicators & Market Knowledge**

On Day 2, the focus is on learning **trading indicators** and acquiring foundational **market knowledge** for various asset classes including **Equities**, **Foreign Exchange (FX)**, **Futures**, **Options**, and **Interest Rates**. The section will cover popular trading indicators, mathematical/statistical formulas, and practical implementation in both Python3 and C++20 using popular third-party libraries.

<div align="right"><a href="../README.md" target="_blacnk"><img src="https://img.shields.io/badge/Back To Full Course-blue?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

---

### **Day-2 Plan Overview** <a name="top"></a>

1. [**1. Moving Averages (MA)**](#1-moving-averages-ma)
2. [**2. Relative Strength Index (RSI)**](#2-relative-strength-index-rsi)
3. [**3. Bollinger Bands**](#3-bollinger-bands)
4. [**4. Market Knowledge: Asset Classes**](#4-market-knowledge-asset-classes)
5. [**Questions and Solutions Section**](#questions-and-solutions-section)

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

### **1. Moving Averages (MA)**

#### **Key Concepts and Mathematical Formulae**

1. **Simple Moving Average (SMA)**:
- The SMA of a time series $$X_t$$ over a period $$N$$ is defined as:

$$
SMA_t = \frac{1}{N} \sum_{i=0}^{N-1} X_{t-i}
$$

- It smooths out short-term fluctuations in prices.

2. **Exponential Moving Average (EMA)**:
- The EMA gives more weight to recent data points:

$$
EMA_t = \alpha \cdot X_t + (1 - \alpha) \cdot EMA_{t-1}
$$

where $$\alpha = \frac{2}{N+1}$$ is the smoothing factor.

#### **Code Implementation: SMA and EMA**

- **Python (Pandas)**:
    ```python
    import pandas as pd

    # Load the data (example with a stock price CSV)
    data = pd.read_csv('stock_data.csv')

    # Calculate the SMA
    data['SMA_20'] = data['Close'].rolling(window=20).mean()

    # Calculate the EMA
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()

    print(data[['Close', 'SMA_20', 'EMA_20']].tail())
    ```

- **C++20 (ta-lib C++ wrapper)**:
    ```cpp
    #include <iostream>
    #include <ta-lib/ta_libc.h>
    
    int main() {
        // Example close prices (for a stock)
        double closePrices[100] = { /*... load prices here ...*/ };

        // Variables for output
        double sma[100], ema[100];
        int outBeg, outNb;

        // Calculate the 20-period SMA
        TA_SMA(0, 99, closePrices, 20, &outBeg, &outNb, sma);

        // Calculate the 20-period EMA
        TA_EMA(0, 99, closePrices, 20, &outBeg, &outNb, ema);

        // Print results
        std::cout << "SMA and EMA calculated for the last 5 periods:" << std::endl;
        for (int i = 95; i < 100; ++i) {
            std::cout << "SMA: " << sma[i] << " EMA: " << ema[i] << std::endl;
        }

        return 0;
    }
    ```

---

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

### **2. Relative Strength Index (RSI)**

#### **Key Concepts and Mathematical Formulae**

1. **RSI Formula**:
The RSI measures the speed and change of price movements:

$$
RSI = 100 - \left(\frac{100}{1 + RS}\right)
$$

where $$RS = \frac{\text{Average Gain}}{\text{Average Loss}}$$ over a given period.

2. **RSI Interpretation**:
- RSI above 70 indicates overbought conditions (a potential sell signal).
- RSI below 30 indicates oversold conditions (a potential buy signal).

#### **Code Implementation: RSI**

- **Python (TA-Lib)**:
    ```python
    import talib
    import numpy as np

    # Load closing prices as a NumPy array
    close_prices = np.array(data['Close'])

    # Calculate the RSI for a 14-period window
    rsi = talib.RSI(close_prices, timeperiod=14)

    print(f"RSI values for the last 5 periods: {rsi[-5:]}")
    ```

- **C++20 (ta-lib C++ wrapper)**:
    ```cpp
    #include <iostream>
    #include <ta-lib/ta_libc.h>

    int main() {
        // Example closing prices for a stock
        double closePrices[100] = { /*... load prices here ...*/ };

        // Variables for output
        double rsi[100];
        int outBeg, outNb;

        // Calculate the 14-period RSI
        TA_RSI(0, 99, closePrices, 14, &outBeg, &outNb, rsi);

        // Print last 5 RSI values
        std::cout << "RSI values for the last 5 periods:" << std::endl;
        for (int i = 95; i < 100; ++i) {
            std::cout << rsi[i] << std::endl;
        }

        return 0;
    }
    ```

---

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

### **3. Bollinger Bands**

#### **Key Concepts and Mathematical Formulae**

1. **Bollinger Bands** consist of three lines:
   - **Middle Band**: $$SMA_t$$, the simple moving average over a certain period.
   - **Upper Band**: $$Upper_t = SMA_t + k \cdot \sigma_t$$, where $$k$$ is a constant (usually 2) and $$\sigma_t$$ is the standard deviation of the asset’s price over the same period.
   - **Lower Band**: $$Lower_t = SMA_t - k \cdot \sigma_t$$.

#### **Code Implementation: Bollinger Bands**

- **Python (TA-Lib)**:
    ```python
    import talib

    # Load closing prices as a NumPy array
    close_prices = np.array(data['Close'])

    # Calculate Bollinger Bands for a 20-period window
    upperband, middleband, lowerband = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    print(f"Upper Band: {upperband[-5:]}")
    print(f"Middle Band: {middleband[-5:]}")
    print(f"Lower Band: {lowerband[-5:]}")
    ```

- **C++20 (ta-lib C++ wrapper)**:
    ```cpp
    #include <iostream>
    #include <ta-lib/ta_libc.h>

    int main() {
        // Example closing prices for a stock
        double closePrices[100] = { /*... load prices here ...*/ };

        // Variables for output
        double upperband[100], middleband[100], lowerband[100];
        int outBeg, outNb;

        // Calculate the 20-period Bollinger Bands
        TA_BBANDS(0, 99, closePrices, 20, 2, 2, TA_MAType_SMA, &outBeg, &outNb, upperband, middleband, lowerband);

        // Print last 5 Bollinger Band values
        std::cout << "Upper Band: " << upperband[95] << ", Middle Band: " << middleband[95] << ", Lower Band: " << lowerband[95] << std::endl;

        return 0;
    }
    ```

---

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

### **4. Market Knowledge: Asset Classes**

#### **1. Equities**
- **Definition**: Stocks represent ownership in a company. Shareholders earn returns through dividends and capital gains.
- **Key Metrics**: Price-to-Earnings (P/E) ratio, Dividend Yield.
- **Formulae**:
  - **Dividend Yield**: $$\text{Dividend Yield} = \frac{\text{Annual Dividend}}{\text{Stock Price}}$$
  - **P/E Ratio**: $$\text{P/E Ratio} = \frac{\text{Stock Price}}{\text{Earnings per Share}}$$

#### **2. Foreign Exchange (FX)**
- **Definition**: The global market for trading currencies.

- **Key Metrics**: Exchange rate, Pips, Spread.

- **Formula**: The exchange rate is expressed as:

$$
\text{Rate}_{\text{USD/EUR}} = \frac{1}{\text{Rate}_{\text{EUR/USD}}}
$$

#### **3. Futures**
   - **Definition**: Financial contracts obligating the buyer to purchase an asset or the seller to sell an asset at a predetermined future date and price.
   - **Key Metrics**: Margin, Mark-to-Market (MTM), Leverage.
   - **Formula**:
     $$
     \text{Leverage} = \frac{\text{Position Value}}{\text{Initial Margin Requirement}}
     $$

#### **4. Options**
   - **Definition**: Contracts that give the holder the right, but not the obligation, to buy (call) or sell (put) an asset at a specific price before a specific date.
   - **Key Metrics**: Strike Price, Option Premium, Greeks (Delta, Gamma, Theta, Vega).
   - **Formula**: **Delta**:
     $$
     \Delta = \frac{\partial V}{\partial S}
     $$
     where $$V$$ is the option price and $$S$$ is the underlying asset price.

#### **5. Interest Rates**
   - **Definition**: The cost of

 borrowing money or the return for lending money.
   - **Key Metrics**: Yield Curve, Federal Funds Rate, LIBOR.

---

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

### **Questions and Solutions Section**

#### **Question 1**: Calculate a 20-period SMA and EMA for given stock prices. Show the final values of SMA and EMA for the last 5 periods.

- **Solution**:
  - Python:
    ```python
    import pandas as pd
    import numpy as np

    # Example closing prices
    data = pd.DataFrame({'Close': np.random.random(100)})

    # Calculate SMA and EMA
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()

    print(data[['Close', 'SMA_20', 'EMA_20']].tail())
    ```

  - C++:
    ```cpp
    #include <iostream>
    #include <ta-lib/ta_libc.h>

    int main() {
        // Example closing prices for a stock
        double closePrices[100] = { /*... load prices here ...*/ };

        // Variables for output
        double sma[100], ema[100];
        int outBeg, outNb;

        // Calculate the 20-period SMA
        TA_SMA(0, 99, closePrices, 20, &outBeg, &outNb, sma);

        // Calculate the 20-period EMA
        TA_EMA(0, 99, closePrices, 20, &outBeg, &outNb, ema);

        // Print results
        std::cout << "SMA and EMA for the last 5 periods:" << std::endl;
        for (int i = 95; i < 100; ++i) {
            std::cout << "SMA: " << sma[i] << " EMA: " << ema[i] << std::endl;
        }

        return 0;
    }
    ```

#### **Question 2**: Calculate the RSI for a given stock over a 14-period window.

- **Solution**:
  - Python:
    ```python
    import talib
    import numpy as np

    # Example closing prices as a NumPy array
    close_prices = np.random.random(100)

    # Calculate the RSI for a 14-period window
    rsi = talib.RSI(close_prices, timeperiod=14)

    print(f"RSI values for the last 5 periods: {rsi[-5:]}")
    ```

  - C++:
    ```cpp
    #include <iostream>
    #include <ta-lib/ta_libc.h>

    int main() {
        // Example closing prices for a stock
        double closePrices[100] = { /*... load prices here ...*/ };

        // Variables for output
        double rsi[100];
        int outBeg, outNb;

        // Calculate the 14-period RSI
        TA_RSI(0, 99, closePrices, 14, &outBeg, &outNb, rsi);

        // Print last 5 RSI values
        std::cout << "RSI values for the last 5 periods:" << std::endl;
        for (int i = 95; i < 100; ++i) {
            std::cout << rsi[i] << std::endl;
        }

        return 0;
    }
    ```

This breakdown covers important **trading indicators** and **market knowledge** concepts for Day 2, with clear mathematical formulae and working code examples in both Python3 and C++20.

<div align="right"><a href="../DAY_3/README.md" target="_blacnk"><img src="https://img.shields.io/badge/Proceed To Day 3-green?style=for-the-badge&logo=expo&logoColor=white" /></a></div>
