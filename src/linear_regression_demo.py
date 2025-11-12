"""Linear Regression Demo (Univariate)
Generates synthetic linear data with noise, fits using gradient descent,
prints learned parameters and saves two plots:
 1. data_scatter_fit.png - scatter of data with learned line
 2. cost_history.png - cost vs iteration
Run: python src/linear_regression_demo.py
"""
import os
import random
import math
import matplotlib.pyplot as plt

# Reproducibility
random.seed(42)


def generate_synthetic_data(m, true_w, true_b, noise_std):
    """Generate y = true_w * x + true_b + noise."""
    x = []
    y = []
    for i in range(m):
        xi = -5 + (10 / (m - 1)) * i  # linspace equivalent
        noise = random.gauss(0, noise_std)
        yi = true_w * xi + true_b + noise
        x.append(xi)
        y.append(yi)
    return x, y, true_w, true_b


def compute_cost(x, y, w, b):
    """Compute MSE cost J(w,b) = (1/(2m)) * sum((w*x_i + b - y_i)^2)"""
    m = len(x)
    total_error_sq = 0.0
    for i in range(m):
        prediction = w * x[i] + b
        error = prediction - y[i]
        total_error_sq += error ** 2
    return total_error_sq / (2 * m)


def compute_gradients(x, y, w, b):
    """Compute dw and db: dw = (1/m) * sum((w*x_i + b - y_i) * x_i), db = (1/m) * sum(w*x_i + b - y_i)"""
    m = len(x)
    dw = 0.0
    db = 0.0
    for i in range(m):
        prediction = w * x[i] + b
        error = prediction - y[i]
        dw += error * x[i]
        db += error
    dw /= m
    db /= m
    return dw, db


def gradient_descent(x, y, alpha, iterations):
    """Run gradient descent to find w and b."""
    w = 0.0
    b = 0.0
    cost_history = []
    for it in range(iterations):
        cost = compute_cost(x, y, w, b)
        cost_history.append(cost)
        dw, db = compute_gradients(x, y, w, b)
        w -= alpha * dw
        b -= alpha * db
        # Early stopping if cost change is tiny
        if it > 0 and abs(cost_history[-2] - cost) < 1e-12:
            break
    return w, b, cost_history


def predict(x, w, b):
    """Predict y for given x: y = w * x + b"""
    predictions = []
    for xi in x:
        predictions.append(w * xi + b)
    return predictions


def save_plots(x, y, w, b, cost_history, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Plot data + fitted line
    plt.figure(figsize=(6,4))
    plt.scatter(x, y, color="royalblue", alpha=0.7, label="data")
    x_line = []
    x_min = min(x)
    x_max = max(x)
    for i in range(200):
        xi = x_min + (x_max - x_min) / 199 * i
        x_line.append(xi)
    y_line = predict(x_line, w, b)
    plt.plot(x_line, y_line, color="darkorange", label="fitted line")
    plt.title("Linear Regression Fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    fit_path = os.path.join(out_dir, "data_scatter_fit.png")
    plt.savefig(fit_path, dpi=120)
    plt.close()

    # Plot cost history
    plt.figure(figsize=(6,4))
    plt.plot(cost_history[50:], color="purple")
    plt.title("Cost vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(w,b)")
    plt.tight_layout()
    cost_path = os.path.join(out_dir, "cost_history.png")
    plt.savefig(cost_path, dpi=120)
    plt.close()

    print(f"Saved plots:\n  {fit_path}\n  {cost_path}")


def main():
    # Generate synthetic data
    x, y, true_w, true_b = generate_synthetic_data(m=120, true_w=3.2, true_b=-0.7, noise_std=1.2)
    print(f"True parameters: w={true_w:.3f}, b={true_b:.3f}")

    # Run gradient descent
    alpha = 0.01
    iterations = 1500
    w, b, cost_history = gradient_descent(x, y, alpha, iterations)
    print(f"Learned parameters: w={w:.3f}, b={b:.3f}")
    print(f"Final cost: {cost_history[-1]:.4f} (after {len(cost_history)} iterations)")

    # Save plots
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    save_plots(x, y, w, b, cost_history, out_dir)

    # Show a simple prediction example
    example_x = [0.0, 2.0, -3.0]
    preds = predict(example_x, w, b)
    for xv, yv in zip(example_x, preds):
        print(f"f({xv: .1f}) = {yv: .3f}")


if __name__ == "__main__":
    main()
