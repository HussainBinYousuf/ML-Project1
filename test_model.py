import csv
import numpy as np
from LassoHomotopy.model import LassoHomotopyModel

def main():
    # CSV file path:
    csv_file = "regression_data.csv"

    # 1. Load data from CSV
    rows = []
    with open(csv_file, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(row)

    # 2. Build X (features) and y (target) as NumPy arrays
    X = np.array([
        [float(value) for key, value in r.items() if key.startswith("x_")]
        for r in rows
    ], dtype=float)
    y = np.array([float(r["y"]) for r in rows], dtype=float)

    # 3. Instantiate the LASSO (OLS in this case) model with alpha=0.0
    model = LassoHomotopyModel(alpha=0.0, max_iter=5000, tol=1e-5)
    results = model.fit(X, y)

    # 4. Print the learned coefficients and intercept
    print("Learned coefficients:", results.coef_)
    print("Learned intercept:", results.intercept_)

    # 5. Compare predictions to actual values
    preds = results.predict(X)
    for i in range(min(5, len(X))):
        print(f"X={X[i]} => Predicted={preds[i]:.4f}, Actual={y[i]:.4f}")

    # 6. Compute and print Mean Absolute Error (MAE) and Mean Squared Error (MSE)
    mae = np.mean(np.abs(preds - y))
    mse = np.mean((preds - y)**2)
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")

    # (Optional) Check correlations among features and target
    if X.shape[1] == 2:  # Only if 2 features
        corr_x0_x1 = np.corrcoef(X[:, 0], X[:, 1])[0, 1]
        corr_x0_y = np.corrcoef(X[:, 0], y)[0, 1]
        corr_x1_y = np.corrcoef(X[:, 1], y)[0, 1]
        print("\nAdditional Info:")
        print("Correlation(x0, x1) =", corr_x0_x1)
        print("Correlation(x0, y)  =", corr_x0_y)
        print("Correlation(x1, y)  =", corr_x1_y)

if __name__ == "__main__":
    main()