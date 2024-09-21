# Day 4: Market Knowledge and Data Analysis

#### Overview
On Day 4, we will deepen our understanding of market knowledge and data analysis, focusing on asset classes, financial data sources, data cleaning and preprocessing, exploratory data analysis (EDA), and statistical techniques used in finance.

---

### **1. Market Knowledge: Asset Classes**

#### **1.1 Equities**
Equities represent ownership in a company. Key metrics include:
- **Price-to-Earnings (P/E) Ratio**:
```math
\text{P/E} = \frac{\text{Market Price per Share}}{\text{Earnings per Share}}
```

- **Dividend Yield**:
```math
\text{Dividend Yield} = \frac{\text{Annual Dividends per Share}}{\text{Market Price per Share}}
```

#### **1.2 Forex (FX)**
Forex trading involves currency pairs. Key metrics:
- **Exchange Rate**: The value of one currency for the purpose of conversion to another.
- **Pip**: The smallest price move that a given exchange rate can make based on market convention.

#### **1.3 Options**
Options give the right, but not the obligation, to buy/sell an asset at a predetermined price.
- **Black-Scholes Model**:
```math
C = S_0 N(d_1) - X e^{-rT} N(d_2)
```
Where:
- $$C$$: Call option price
- $$S_0$$: Current stock price
- $$X$$: Strike price
- $$r$$: Risk-free interest rate
- $$T$$: Time to expiration
- $$N(d)$$: Cumulative distribution function of the standard normal distribution
- $$d_1 = \frac{\ln(S_0/X) + (r + \sigma^2/2)T}{\sigma \sqrt{T}}$$
- $$d_2 = d_1 - \sigma \sqrt{T}$$

#### **1.4 Futures and Commodities**
Futures contracts obligate the buyer to purchase an asset at a predetermined price at a specified time in the future.

#### **1.5 Interest Rates**
Interest rates are the cost of borrowing money, expressed as a percentage. They can be influenced by central bank policies, inflation, and economic growth.

---

### **2. Data Sources and Collection**

#### **2.1 Financial Data Sources**
- **Yahoo Finance**: Provides historical stock price data.
- **Quandl**: Offers various datasets, including economic indicators.
- **Bloomberg Terminal**: Comprehensive financial data provider (subscription-based).

#### **2.2 Data Retrieval Example**

**Python Code Example** (Using `yfinance`):
```python
import yfinance as yf

# Download historical data for Apple Inc.
data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")
print(data.head())
```

**C++ Code Example**:
Using a library like **libcurl** to make HTTP requests for data retrieval.
```cpp
#include <iostream>
#include <curl/curl.h>
#include <string>

size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

int main() {
    CURL* curl;
    CURLcode res;
    std::string readBuffer;

    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();
    
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, "https://query1.finance.yahoo.com/v7/finance/download/AAPL?period1=1577836800&period2=1672537600&interval=1d&events=history");
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        
        std::cout << readBuffer << std::endl;
    }
    
    curl_global_cleanup();
    return 0;
}
```

---

### **3. Data Cleaning and Preprocessing**

#### **3.1 Common Data Cleaning Steps**
- Handling missing values (imputation or removal).
- Removing duplicates.
- Converting data types (e.g., date parsing).

**Python Code Example**:
```python
import pandas as pd

# Load the dataset
data = pd.read_csv('AAPL.csv')

# Handle missing values
data.fillna(method='ffill', inplace=True)  # Forward fill
data.drop_duplicates(inplace=True)

# Convert date column to datetime
data['Date'] = pd.to_datetime(data['Date'])
print(data.info())
```

**C++ Code Example**:
Using the **Armadillo** library for handling data:
```cpp
#include <armadillo>

int main() {
    arma::mat data;
    data.load("AAPL.csv", arma::csv_ascii);
    
    // Handling missing values and duplicates can be manually implemented here
    // Armadillo does not have direct methods for this, as it's primarily for matrix operations.
    
    return 0;
}
```

---

### **4. Exploratory Data Analysis (EDA)**

#### **4.1 Visualization**
Visualizing data is crucial for understanding patterns and trends.

**Python Code Example**:
```python
import matplotlib.pyplot as plt

# Plot the closing price
plt.figure(figsize=(10, 5))
plt.plot(data['Date'], data['Close'], label='Close Price')
plt.title('AAPL Closing Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
```

**C++ Code Example**:
Using a library like **matplotplusplus** for plotting:
```cpp
#include <matplot/matplot.h>

int main() {
    std::vector<double> dates = {/* Date data */};
    std::vector<double> close_prices = {/* Closing prices */};

    matplot::figure_size(10, 5);
    matplot::plot(dates, close_prices);
    matplot::title("AAPL Closing Price");
    matplot::xlabel("Date");
    matplot::ylabel("Price");
    matplot::show();

    return 0;
}
```

---

### **5. Statistical Techniques in Finance**

#### **5.1 Descriptive Statistics**
- **Mean**:
```math
\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
```
- **Standard Deviation**:
```math
\sigma = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2}
```

**Python Code Example**:
```python
mean = data['Close'].mean()
std_dev = data['Close'].std()
print(f'Mean: {mean}, Standard Deviation: {std_dev}')
```

**C++ Code Example**:
```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>

double mean(const std::vector<double>& data) {
    return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

double std_dev(const std::vector<double>& data, double mean_value) {
    double variance = 0.0;
    for (const auto& value : data) {
        variance += std::pow(value - mean_value, 2);
    }
    return std::sqrt(variance / (data.size() - 1));
}

int main() {
    std::vector<double> closing_prices = {/* Closing prices */};
    
    double mean_value = mean(closing_prices);
    double std_dev_value = std_dev(closing_prices, mean_value);

    std::cout << "Mean: " << mean_value << ", Standard Deviation: " << std_dev_value << std::endl;
    return 0;
}
```

---

### **Question & Answer Section**

#### **Q1: What is the purpose of the P/E ratio, and how is it calculated?**
**A1:** The P/E ratio measures a company's current share price relative to its per-share earnings. It is calculated as:
```math
\text{P/E} = \frac{\text{Market Price per Share}}{\text{Earnings per Share}}
```

#### **Q2: How do you handle missing values in a dataset?**
**A2:** Missing values can be handled by removing them, filling them with mean/median values, or using forward/backward filling.

#### **Q3: Write a Python function to calculate the mean and standard deviation of a financial time series.**
**A3:**
```python
def calculate_statistics(series):
    mean = series.mean()
    std_dev = series.std()
    return mean, std_dev

mean, std_dev = calculate_statistics(data['Close'])
print(f'Mean: {mean}, Standard Deviation: {std_dev}')
```

#### **Q4: Discuss how you would visualize the performance of a stock over time.**
**A4:** I would plot the stock's closing prices against time using line charts to observe trends, identify peaks and troughs, and analyze the overall performance visually.

---

### Conclusion
By the end of Day 4, you should have a solid foundation in market knowledge, data collection, cleaning, and exploratory data analysis. This knowledge will be crucial as you proceed to the next phase of your learning journey, focusing on **Portfolio Optimization and Risk Management**.
