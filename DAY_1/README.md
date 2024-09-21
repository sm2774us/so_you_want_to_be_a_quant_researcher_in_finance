### **Day 1: Mathematics & Statistics for Quantitative Finance (Detailed Breakdown)**

This detailed breakdown of Day 1 will provide mathematical formulae, proofs, explanations, and code implementations in both Python 3 and C++20 using popular third-party libraries. The day will be split into fundamental sections with a focus on mathematical and statistical theory, code examples, and finally, questions and solutions to test understanding.

<div align="right"><a href="../README.md" target="_blacnk"><img src="https://img.shields.io/badge/Back To Full Course-blue?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

---

### **1. Linear Algebra for Quantitative Finance**

#### **Key Concepts and Mathematical Formulae**

1. **Vectors and Matrices**:
- **Vector**:

$$
\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}
$$

  Operations: dot product, cross product.

- **Matrix multiplication**:

$$
C = AB, \text{ where } C_{ij} = \sum_{k} A_{ik} B_{kj}
$$

- **Inverse of a matrix** $A^{-1}$:

$$
A^{-1} A = I, \text{ where } I \text{ is the identity matrix.}
$$

2. **Eigenvalues and Eigenvectors**:
- Given a matrix $$A$$, an eigenvector $$\mathbf{v}$$ and its corresponding eigenvalue $$\lambda$$ satisfy:

$$
A \mathbf{v} = \lambda \mathbf{v}
$$

- The eigenvalue decomposition of a matrix is $$A = V \Lambda V^{-1}$$, where $$V$$ contains the eigenvectors and $$\Lambda$$ contains the eigenvalues.

3. **Singular Value Decomposition (SVD)**:
- Any matrix $$A$$ can be decomposed as:

$$
A = U \Sigma V^T
$$

where $$U$$ and $$V$$ are orthogonal matrices, and $$\Sigma$$ is a diagonal matrix of singular values.

#### **Code Implementation: Eigenvalues & Eigenvectors**

- **Python (NumPy)**:
    ```python
    import numpy as np

    # Matrix A
    A = np.array([[4, 2], 
                  [1, 3]])

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)

    print("Eigenvalues:")
    print(eigenvalues)
    print("Eigenvectors:")
    print(eigenvectors)
    ```

- **C++20 (Eigen)**:
    ```cpp
    #include <iostream>
    #include <Eigen/Dense>

    int main() {
        Eigen::Matrix2d A;
        A << 4, 2, 
             1, 3;

        Eigen::EigenSolver<Eigen::Matrix2d> solver(A);

        std::cout << "Eigenvalues:\n" << solver.eigenvalues() << std::endl;
        std::cout << "Eigenvectors:\n" << solver.eigenvectors() << std::endl;
        
        return 0;
    }
    ```

---

### **2. Probability and Statistics**

#### **Key Concepts and Mathematical Formulae**

1. **Expected Value**:

$$
\mathbb{E}[X] = \sum_{i} x_i P(x_i)
$$

  For continuous distributions:

$$
\mathbb{E}[X] = \int_{-\infty}^{\infty} x f(x) \, dx
$$

   The expected value measures the central tendency or average of a random variable.

2. **Variance and Standard Deviation**:
- **Variance**:

$$
\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]
$$

- **Standard Deviation**:
  $$\sigma(X) = \sqrt{\text{Var}(X)}$$

3. **Covariance**:

$$
\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]
$$

  Covariance indicates how two random variables move together.

4. **Normal Distribution**:
The probability density function (PDF) of a normal distribution is:

$$
f(x) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

Used to model asset returns.

#### **Code Implementation: Normal Distribution & Statistics**

- **Python (NumPy & SciPy)**:
    ```python
    import numpy as np
    from scipy.stats import norm

    # Parameters
    mu = 0
    sigma = 1

    # Generate random samples
    data = np.random.normal(mu, sigma, 1000)

    # Compute mean and variance
    mean = np.mean(data)
    variance = np.var(data)

    print(f"Mean: {mean}, Variance: {variance}")

    # Probability density function
    x = np.linspace(-3, 3, 100)
    pdf = norm.pdf(x, mu, sigma)

    print(f"PDF values: {pdf[:5]}")
    ```

- **C++20 (Boost Random)**:
    ```cpp
    #include <iostream>
    #include <boost/random.hpp>
    #include <boost/math/distributions/normal.hpp>

    int main() {
        boost::random::mt19937 rng;
        boost::random::normal_distribution<> dist(0, 1);

        // Generate random samples
        double sum = 0;
        double sum_sq = 0;
        int N = 1000;
        for (int i = 0; i < N; ++i) {
            double sample = dist(rng);
            sum += sample;
            sum_sq += sample * sample;
        }

        double mean = sum / N;
        double variance = (sum_sq / N) - (mean * mean);

        std::cout << "Mean: " << mean << ", Variance: " << variance << std::endl;

        return 0;
    }
    ```

---

### **3. Calculus and Stochastic Calculus**

#### **Key Concepts and Mathematical Formulae**

1. **Derivatives**:
- First derivative $$f'(x)$$ represents the rate of change of a function.

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

- Second derivative $$f''(x)$$ represents the curvature or concavity of the function.

2. **Ito's Lemma**:
- In stochastic calculus, Ito’s Lemma is a fundamental result used to differentiate stochastic processes.
- If $$f(X_t)$$ is a function of a stochastic process $$X_t$$, then Ito's Lemma gives:

$$
df(X_t) = \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial X_t} dX_t + \frac{1}{2} \frac{\partial^2 f}{\partial X_t^2} (dX_t)^2
$$

- This is used to derive the **Black-Scholes equation**.

#### **Code Implementation: Differentiation**

- **Python (SymPy)**:
    ```python
    from sympy import symbols, diff

    # Define the variables
    S, t, sigma = symbols('S t sigma')

    # Option price formula
    option_price = S * sigma * t

    # Compute Delta (first derivative w.r.t. S)
    delta = diff(option_price, S)
    print(f"Delta: {delta}")
    ```

- **C++20**: For differentiation, numerical methods can be implemented manually or using a library like AutoDiff.

---

### **4. Time Series Analysis**

#### **Key Concepts and Mathematical Formulae**

1. **Autoregressive (AR) Model**:
- The AR(1) model is given by:

$$
X_t = \phi X_{t-1} + \epsilon_t
$$

- $$\epsilon_t$$ is white noise with mean zero and variance $$\sigma^2$$.
  
2. **Moving Average (MA) Model**:
- The MA(1) model:

$$
X_t = \mu + \epsilon_t + \theta \epsilon_{t-1}
$$

3. **ARIMA**:
- The ARIMA model generalizes AR and MA by integrating the differencing term $$I$$:

$$
X_t = \phi_1 X_{t-1} + \dots + \phi_p X_{t-p} + \theta_1 \epsilon_{t-1} + \dots + \theta_q \epsilon_{t-q} + d
$$

#### **Code Implementation: ARIMA Model**

- **Python (Statsmodels)**:
    ```python
    import pandas as pd
    from statsmodels.tsa.arima.model import ARIMA

    # Load dataset
    data = pd.read_csv('stock_data.csv')

    # Fit ARIMA model
    model = ARIMA(data['Close'], order=(1, 1, 1))
    model_fit = model.fit()

    print(model_fit.summary())
    ```

- **C++**: Implementing ARIMA requires custom code or using a third

-party library (e.g., dlib or Eigen).

---

<div align="right"><a href="#top" target="_blacnk"><img src="https://img.shields.io/badge/Back To Top-orange?style=for-the-badge&logo=expo&logoColor=white" /></a></div>

### **Questions and Solutions Section**

#### **Question 1**: Eigenvalues of a Covariance Matrix
- Given the covariance matrix:

$$
\Sigma = \begin{bmatrix} 1 & 0.5 \\ 0.5 & 1 \end{bmatrix}
$$

  Calculate the eigenvalues and eigenvectors.

- **Solution**:
  - Python:
    ```python
    import numpy as np
    Sigma = np.array([[1, 0.5], [0.5, 1]])
    eigenvalues, eigenvectors = np.linalg.eig(Sigma)
    print(eigenvalues)
    print(eigenvectors)
    ```
  - C++:
    ```cpp
    #include <Eigen/Dense>
    int main() {
        Eigen::Matrix2d Sigma;
        Sigma << 1, 0.5, 0.5, 1;
        Eigen::EigenSolver<Eigen::Matrix2d> solver(Sigma);
        std::cout << "Eigenvalues:\n" << solver.eigenvalues() << std::endl;
    }
    ```

#### **Question 2**: Fit a Normal Distribution and Calculate the Mean
- Fit a normal distribution to a random dataset and compute the mean and variance.

- **Solution**:
  - Python:
    ```python
    import numpy as np
    data = np.random.normal(0, 1, 1000)
    mean = np.mean(data)
    variance = np.var(data)
    print(f"Mean: {mean}, Variance: {variance}")
    ```
  - C++:
    ```cpp
    #include <boost/random.hpp>
    #include <iostream>
    int main() {
        boost::random::mt19937 rng;
        boost::random::normal_distribution<> dist(0, 1);
        double sum = 0;
        double sum_sq = 0;
        for (int i = 0; i < 1000; ++i) {
            double sample = dist(rng);
            sum += sample;
            sum_sq += sample * sample;
        }
        double mean = sum / 1000;
        double variance = (sum_sq / 1000) - (mean * mean);
        std::cout << "Mean: " << mean << ", Variance: " << variance << std::endl;
    }
    ```

This concludes Day 1, providing a solid foundation in the mathematics and statistics required for quantitative finance with practical examples in both __`Python 3`__ and __`C++20`__.

<div align="right"><a href="../README.md" target="_blacnk"><img src="https://img.shields.io/badge/Back To Full Course-blue?style=for-the-badge&logo=expo&logoColor=white" /></a></div>
