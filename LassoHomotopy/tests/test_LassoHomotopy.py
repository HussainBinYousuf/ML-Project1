import csv
import numpy as np
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Adjust the import to match where your LassoHomotopyModel is located
from LassoHomotopy.model import LassoHomotopyModel

def test_predict(threshold=5.0):
    """
    This is a PyTest test function that:
      1. Loads data from a CSV file.
      2. Fits a LassoHomotopyModel.
      3. Prints out some predictions and actuals.
      4. Computes the MSE.
      5. Asserts the MSE is below a user-defined threshold.
    """

    # 1. Specify CSV file path
    csv_file = "LassoHomotopy/tests/small_test.csv"

    # 2. Read data from CSV
    rows = []
    with open(csv_file, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(row)

    # 3. Build X (features) and y (targets) as NumPy arrays
    X = np.array([
        [float(value) for key, value in r.items() if key.startswith("x_")]
        for r in rows
    ], dtype=float)
    y = np.array([float(r["y"]) for r in rows], dtype=float)

    # 4. Instantiate and fit the LassoHomotopyModel
    #    (Adjust parameters as needed. If your model doesn't accept 'alpha', remove/rename it.)
    model = LassoHomotopyModel(alpha=0.0, max_iter=5000, tol=1e-5)
    results = model.fit(X, y)

    # 5. Make predictions on the same data (training set) for demonstration
    preds = results.predict(X)

    # Print some info - PyTest will capture this in the test output
    print("\n=== LassoHomotopyModel Test ===")
    print("Coefficients:", results.coef_)
    print("Intercept:", results.intercept_)
    for i in range(min(5, len(X))):
        print(f"Sample {i} => Predicted={preds[i]:.4f}, Actual={y[i]:.4f}")

    # 6. Compute the Mean Squared Error (MSE)
    mse = np.mean((preds - y)**2)
    print(f"\nMSE: {mse:.4f}")

    # 7. Assert that the MSE is below your chosen threshold
    assert mse < threshold, f"MSE={mse:.4f} exceeds threshold={threshold}"
