# Linear Regression Intro (Python + Markdown)

This mini-project introduces univariate linear regression with a concise Markdown explanation and a pure-Python demo script that generates data, runs gradient descent, and saves plots. No Jupyter required.

## Files
- `docs/linear_regression_intro.md` — Conceptual introduction: dataset notation, model, cost, derivatives, gradient descent.
- `src/linear_regression_demo.py` — Standalone script: generates synthetic data, fits a line with gradient descent, and saves figures to `outputs/`.
- `outputs/` — Created at runtime with:
  - `data_scatter_fit.png`: data points and fitted line
  - `cost_history.png`: cost vs. iteration

## Quick start
1. (Optional) Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies
```bash
python -m pip install --upgrade pip
pip install numpy matplotlib
```

3. Run the demo
```bash
python src/linear_regression_demo.py
```

You should see console output with true and learned parameters, plus PNG files saved under `outputs/`.

## Teaching tips
- Encourage students to derive the gradients by hand before running the code.
- Try different learning rates (alpha) and iteration counts to show effects on convergence.
- Modify the noise level to see how it impacts the final fit.
