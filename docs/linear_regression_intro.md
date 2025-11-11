# Introduction to Linear Regression

Linear regression is a foundational supervised learning method for modeling the relationship between an input variable and a real-valued output. In its simplest (univariate) form, we assume the relationship between an input x and target y is approximately linear.

## Dataset and notation
- We observe a training set of m examples:  
  $\mathcal{D} = \{(x^{(i)},\, y^{(i)})\}_{i=1}^m$ where $x^{(i)} \in \mathbb{R}$ and $y^{(i)} \in \mathbb{R}$.
- For the univariate case (one feature), $x^{(i)}$ is a scalar. For the multivariate case with n features, $x^{(i)} \in \mathbb{R}^n$.

## Model (hypothesis)
For the univariate case, our model (also called the hypothesis) is a straight line:

$$
\begin{aligned}
 f_{w,b}(x) &= w\,x + b
\end{aligned}
$$

- $w$ is the slope (weight), and $b$ is the intercept (bias).
- Multivariate generalization: with $x \in \mathbb{R}^n$ and $w \in \mathbb{R}^n$,

$$
 f_{w,b}(x) = w^\top x + b.
$$

## Cost function (Mean Squared Error)
We measure how well a particular $(w,b)$ fits the training data using the Mean Squared Error (MSE) cost function:

$$
\begin{aligned}
J(w,b) &= \frac{1}{2m} \sum_{i=1}^{m} \Big(f_{w,b}(x^{(i)}) - y^{(i)}\Big)^2 \\
       &= \frac{1}{2m} \sum_{i=1}^{m} \Big( w\,x^{(i)} + b - y^{(i)} \Big)^2.
\end{aligned}
$$

- The factor $\tfrac{1}{2}$ is included so that derivatives are slightly cleaner.

## Derivatives (gradients)
To minimize $J(w,b)$, we compute its partial derivatives with respect to $w$ and $b$.

Univariate derivatives:

$$
\begin{aligned}
\frac{\partial J}{\partial w} &= \frac{1}{m} \sum_{i=1}^{m} \Big( w\,x^{(i)} + b - y^{(i)} \Big)\, x^{(i)} \\
\frac{\partial J}{\partial b} &= \frac{1}{m} \sum_{i=1}^{m} \Big( w\,x^{(i)} + b - y^{(i)} \Big).
\end{aligned}
$$

For multivariate case with $X \in \mathbb{R}^{m\times n}$ (rows $(x^{(i)})^\top$) and $\hat{y} = Xw + b\,\mathbf{1}$, the derivatives generalize to matrix form.

## Gradient Descent
Gradient descent is an iterative optimization algorithm that updates parameters in the direction of the negative gradient to reduce the cost.

Initialize $w$ and $b$ (e.g., to 0), choose a learning rate $\alpha>0$, and repeat for a chosen number of iterations $T$:

$$
\begin{aligned}
 w &\leftarrow w - \alpha\, \frac{\partial J}{\partial w} \\
 b &\leftarrow b - \alpha\, \frac{\partial J}{\partial b}
\end{aligned}
$$

Pseudocode (univariate):

```
input: data {(x^(i), y^(i))}_{i=1}^m, learning rate α, iterations T
initialize: w ← 0, b ← 0
for t in {1,…,T}:
    compute gradients:
        dw = (1/m) * Σ_i (w*x^(i) + b − y^(i)) * x^(i)
        db = (1/m) * Σ_i (w*x^(i) + b − y^(i))
    update:
        w ← w − α * dw
        b ← b − α * db
return w, b
```

### Notes on practice
- Learning rate (α) controls the step size. Too large can diverge; too small can be slow.
- Feature scaling (for multivariate x) often improves convergence.
- Convergence can be monitored by plotting $J(w,b)$ over iterations.

## What you will run next
The companion Python script generates synthetic data from a true line, implements gradient descent using the derivatives above, and saves:
- A scatter plot with the learned regression line.
- A cost-versus-iterations plot showing convergence.
