import csv
import numpy as np
import os
import sys
import pytest
try:
    from sklearn.datasets import make_regression
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Lasso as SklearnLasso
    sklearn_available = True
except ImportError:
    sklearn_available = False

try:
    import pandas as pd
    pandas_available = True
except ImportError:
    pandas_available = False

# Add project root to path so we can import our model
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import our LASSO implementation
from LassoHomotopy.model import LassoHomotopyModel


#from LassoHomotopy.model.LassoHomotopy import LassoHomotopyModel
### newly added starts


def test_highly_collinear_with_missing():
    """
    Combined Test 1: Highly Collinear Data with Missing Values.
    Create nearly identical features (extreme collinearity) and inject a NaN.
    Expect the model to either clean or (more likely) raise a ValueError.
    """
    np.random.seed(42)
    n_samples = 100
    base = np.random.randn(n_samples, 1)
    # Create 3 highly collinear features
    X = np.hstack([base + np.random.randn(n_samples, 1) * 1e-4 for _ in range(3)])
    # Introduce a missing value
    X[10, 1] = np.nan
    y = 3 * base.squeeze() + np.random.randn(n_samples) * 0.1

    model = LassoHomotopyModel(alpha=0.1)
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_high_dimensional_sparse_with_outliers():
    """
    Combined Test 2: High-Dimensional Sparse Data with Outliers.
    Generate a high-dim dataset (p >> n) and force sparsity by zeroing small values.
    Then inject extreme outliers into y.
    """
    np.random.seed(42)
    n_samples = 10
    n_features = 50
    X = np.random.randn(n_samples, n_features)
    # Zero out most small-magnitude values to simulate sparsity
    X[np.abs(X) < 1.5] = 0
    true_coef = np.zeros(n_features)
    true_coef[:3] = [1.0, -2.0, 3.0]
    y = X @ true_coef + np.random.randn(n_samples) * 0.1
    # Inject an extreme outlier in y
    y[0] = 1e5

    model = LassoHomotopyModel(alpha=0.1)
    results = model.fit(X, y)
    preds = results.predict(X)
    assert np.all(np.isfinite(preds)), "Predictions should be finite despite outliers"

def test_mixed_scale_constant_nearconstant():
    """
    Combined Test 3: Mixed Scale Features with Constant and Near-Constant Columns.
    One column is completely constant; another is nearly constant;
    remaining columns are on standard scales.
    Expect constant/near-constant features to have near-zero coefficients.
    """
    np.random.seed(42)
    n_samples = 50
    constant_feature = np.full((n_samples, 1), 100.0)
    near_constant_feature = 99.9 + np.random.randn(n_samples, 1) * 0.001
    random_feature = np.random.randn(n_samples, 1)
    X = np.hstack([constant_feature, near_constant_feature, random_feature])
    # Let y depend only on the random feature
    y = 0 * constant_feature.squeeze() + 0 * near_constant_feature.squeeze() + 2 * random_feature.squeeze() + np.random.randn(n_samples) * 0.1

    model = LassoHomotopyModel(alpha=0.1)
    results = model.fit(X, y)
    preds = results.predict(X)
    assert preds.shape[0] == n_samples, "Predictions shape must match number of samples"
    # Expect constant features to contribute nothing
    assert abs(results.coef_[0]) < 1e-6 and abs(results.coef_[1]) < 1e-6, "Constant and near-constant features should have near-zero coefficients"

def test_mismatched_nonnumeric_outliers():
    """
    Combined Test 4: Mismatched Data with Non-Numeric and Outlier Values.
    Create an X with a non-numeric column (and possibly outliers) and expect an error.
    """
    # Here, one column is non-numeric.
    X = np.array([[1.0, "non-numeric"], [2.0, "3.0"]])
    y = np.array([1.0, 2.0])
    model = LassoHomotopyModel(alpha=0.1)
    with pytest.raises(Exception):
        model.fit(X, y)

def test_convergence_under_stress():
    """
    Combined Test 5: Convergence Under Stress.
    Use an extremely ill-conditioned matrix (high condition number) and
    an extreme regularization parameter. Check that the model converges
    (i.e., produces finite predictions and a finite error).
    """
    np.random.seed(42)
    n_samples, n_features = 100, 10
    # Create an ill-conditioned matrix via SVD
    X_base = np.random.randn(n_samples, n_features)
    U, s, Vt = np.linalg.svd(X_base, full_matrices=False)
    s[1:] = s[1:] / 1000.0  # artificially worsen conditioning
    X = U @ np.diag(s) @ Vt
    true_coef = np.zeros(n_features)
    true_coef[0] = 1.0
    true_coef[3] = -0.5
    y = X @ true_coef + np.random.randn(n_samples) * 0.1

    model = LassoHomotopyModel(alpha=1e4, tol=1e-6, max_iter=200)
    results = model.fit(X, y)
    preds = results.predict(X)
    mse = np.mean((preds - y) ** 2)
    assert np.isfinite(mse), "MSE should be finite even under extreme stress"

def test_sequential_data_changing_dynamics():
    """
    Combined Test 6: Sequential Data with Changing Dynamics.
    Simulate a dataset where the underlying relationship changes midway.
    Although LASSO is a linear model, ensure that the fit produces valid outputs.
    """
    np.random.seed(42)
    n_samples = 200
    n_features = 5
    # First half: relationship y = 2*x0 + noise
    X1 = np.random.randn(n_samples // 2, n_features)
    y1 = 2 * X1[:, 0] + np.random.randn(n_samples // 2) * 0.1
    # Second half: relationship y = -3*x0 + noise
    X2 = np.random.randn(n_samples // 2, n_features)
    y2 = -3 * X2[:, 0] + np.random.randn(n_samples // 2) * 0.1

    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2])
    model = LassoHomotopyModel(alpha=0.1)
    results = model.fit(X, y)
    preds = results.predict(X)
    assert np.all(np.isfinite(preds)), "Predictions should be finite for sequential data with changing dynamics"



def test_reproducibility():
    """Model produces identical results with the same random seed."""
    np.random.seed(42)
    X = np.random.randn(50, 5)
    true_coef = np.array([1.0, 0.0, 2.0, -1.0, 0.5])
    y = X @ true_coef + np.random.randn(50) * 0.1

    model1 = LassoHomotopyModel(alpha=0.1, tol=1e-6)
    results1 = model1.fit(X, y)
    
    # Reset seed and generate data again to ensure reproducibility.
    np.random.seed(42)
    X2 = np.random.randn(50, 5)
    y2 = X2 @ true_coef + np.random.randn(50) * 0.1
    model2 = LassoHomotopyModel(alpha=0.1, tol=1e-6)
    results2 = model2.fit(X2, y2)
    
    np.testing.assert_allclose(results1.coef_, results2.coef_, atol=1e-6)
    np.testing.assert_allclose(results1.predict(X), results2.predict(X), atol=1e-6)

def test_non_numeric_input():
    """Model should raise an error when non-numeric input is provided."""
    X = np.array([["a", "b"], ["c", "d"]])
    y = np.array([1, 2])
    model = LassoHomotopyModel(alpha=0.1)
    with pytest.raises(Exception):
        model.fit(X, y)

def test_no_feature_input():
    """Model should raise an error or handle the case when there are zero features."""
    X = np.empty((10, 0))
    y = np.random.randn(10)
    model = LassoHomotopyModel(alpha=0.1)
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_outlier_sensitivity():
    """Model should handle outliers without crashing."""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = X @ np.array([1.5, -2.0, 0.5]) + np.random.randn(100) * 0.1
    # Introduce a large outlier in y
    y[0] = 1e6
    model = LassoHomotopyModel(alpha=0.1)
    results = model.fit(X, y)
    preds = results.predict(X)
    # Ensure predictions are finite and error is computed (may be high, but not NaN)
    assert np.all(np.isfinite(preds)), "Predictions should be finite even with outliers"

def test_api_compliance():
    """Check that the model output contains expected attributes and methods."""
    np.random.seed(42)
    X = np.random.randn(50, 4)
    y = X @ np.array([2, -1, 0, 1]) + np.random.randn(50) * 0.1
    model = LassoHomotopyModel(alpha=0.1)
    results = model.fit(X, y)
    # Check that attributes exist
    assert hasattr(results, "coef_"), "Model output should have 'coef_' attribute"
    assert hasattr(results, "intercept_"), "Model output should have 'intercept_' attribute"
    assert callable(getattr(results, "predict", None)), "Model output should have a callable 'predict' method"
    preds = results.predict(X)
    assert preds.shape[0] == X.shape[0], "Predictions should match number of samples"

# If your model supports convergence logs, you might add:
def test_monotonic_convergence():
     """Test that the error decreases (or does not increase) over iterations."""
     np.random.seed(42)
     X = np.random.randn(100, 10)
     y = X @ np.random.randn(10) + np.random.randn(100) * 0.1
     model = LassoHomotopyModel(alpha=0.1, tol=1e-6, max_iter=100)
     results = model.fit(X, y)
     # Assume model returns a list of error values per iteration (e.g., results.errors)
     errors = results.errors
     # Check that, after an initial period, errors do not increase
     assert all(earlier >= later for earlier, later in zip(errors[5:], errors[6:])), "Convergence errors should be non-increasing"



def test_empty_dataset():
    """Test model behavior with an empty dataset."""
    X = np.empty((0, 5))
    y = np.empty((0,))
    model = LassoHomotopyModel(alpha=0.1)
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_single_sample():
    """Test model behavior when only a single sample is provided."""
    X = np.array([[1, 2, 3]])
    y = np.array([6])
    model = LassoHomotopyModel(alpha=0.1)
    results = model.fit(X, y)
    pred = results.predict(X)[0]
    # The prediction should be a finite number
    assert np.isfinite(pred), "Prediction for a single sample should be finite"

def test_all_zero_features():
    """Test model on data where all features are zero."""
    X = np.zeros((10, 5))
    y = np.zeros(10)
    model = LassoHomotopyModel(alpha=0.1)
    results = model.fit(X, y)
    # With no signal, coefficients should be (almost) zero
    assert np.allclose(results.coef_, np.zeros(5), atol=1e-8), "Coefficients should be zero when features are all zero"

def test_nan_in_data():
    """Test model behavior with NaN values in the dataset."""
    X = np.array([[1, 2, 3], [4, 5, np.nan]])
    y = np.array([1, 2])
    model = LassoHomotopyModel(alpha=0.1)
    with pytest.raises(ValueError):
        # Expecting the model to raise an error due to NaN values
        model.fit(X, y)

def test_high_dimensional_data():
    """Test model on high-dimensional data (more features than samples)."""
    np.random.seed(42)
    n_samples = 10
    n_features = 50  # High-dimensional scenario
    X = np.random.randn(n_samples, n_features)
    true_coef = np.zeros(n_features)
    # Set a few nonzero coefficients
    true_coef[:3] = [2, -1, 1.5]
    y = X @ true_coef + np.random.randn(n_samples) * 0.1
    model = LassoHomotopyModel(alpha=0.1)
    results = model.fit(X, y)
    mse = np.mean((results.predict(X) - y)**2)
    # Expect a low error if the model is working well
    assert mse < 5.0, "High-dimensional data: Model should fit with reasonably low error"

### newly added ends


def test_highly_collinear_extreme():
    """Test the model on an extremely highly collinear dataset."""
    np.random.seed(42)
    n_samples = 100
    # Create one base feature
    base = np.random.randn(n_samples, 1)
    # Create 5 features that are almost identical to base (with very small noise)
    X = np.hstack([base + np.random.randn(n_samples, 1) * 1e-4 for _ in range(5)])
    # The true response depends on the base feature
    y = 3 * base.squeeze() + np.random.randn(n_samples) * 0.1
    
    model = LassoHomotopyModel(alpha=0.1)
    results = model.fit(X, y)
    nonzero_count = np.sum(np.abs(results.coef_) > 1e-10)
    
    # With extreme collinearity, expect that not all 5 features are assigned nonzero coefficients.
    assert nonzero_count < 5, "With extreme collinearity, the model should select only a subset of features"

def test_constant_feature():
    """Test behavior when one feature is constant."""
    np.random.seed(42)
    n_samples = 50
    # Create two features: one constant, one random
    constant_feature = np.ones((n_samples, 1)) * 3.14
    random_feature = np.random.randn(n_samples, 1)
    X = np.hstack([constant_feature, random_feature])
    # Let y depend only on the random feature
    y = 2 * random_feature.squeeze() + np.random.randn(n_samples) * 0.1
    
    model = LassoHomotopyModel(alpha=0.1)
    results = model.fit(X, y)
    # Check that the constant feature has a coefficient near zero.
    assert np.abs(results.coef_[0]) < 1e-6, "Constant feature should have near-zero coefficient"

def test_mismatched_dimensions():
    """Test that the model raises an error when X and y have mismatched numbers of samples."""
    X = np.random.randn(50, 3)
    y = np.random.randn(40)  # Mismatch: 50 samples vs. 40 responses
    model = LassoHomotopyModel(alpha=0.1)
    
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_extreme_regularization():
    """Test that with extremely high alpha, nearly all coefficients are zero."""
    np.random.seed(42)
    n_samples, n_features = 100, 10
    X = np.random.randn(n_samples, n_features)
    # Create true coefficients with nonzero entries
    true_coef = np.zeros(n_features)
    true_coef[:3] = [2, -1, 1.5]
    y = X @ true_coef + np.random.randn(n_samples) * 0.1
    
    # Set alpha very high
    model = LassoHomotopyModel(alpha=1e6)
    results = model.fit(X, y)
    # Expect nearly all coefficients to be zero due to strong regularization
    assert np.allclose(results.coef_, np.zeros(n_features), atol=1e-3), "With extreme regularization, coefficients should be near zero"

def test_negative_alpha():
    """Test that providing a negative alpha raises an error."""
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = np.random.randn(50)
    
    # Negative alpha is not allowed
    with pytest.raises(ValueError):
        LassoHomotopyModel(alpha=-0.5)


def test_predict(threshold=10.0):
    """
    Basic test for model prediction functionality.
    
    Loads data from CSV if available, otherwise creates synthetic data.
    Fits the model and checks if MSE is below a reasonable threshold.
    """

    # Try to use existing CSV file
    csv_file = "small_test.csv"

    try:
        # Load and parse the CSV
        rows = []
        with open(csv_file, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                rows.append(row)

        # Extract features and target
        X = np.array([
            [float(value) for key, value in r.items() if key.startswith("x_")]
            for r in rows
        ], dtype=float)
        y = np.array([float(r["y"]) for r in rows], dtype=float)

    except FileNotFoundError:
        # No CSV? Let's make our own test data
        print(f"\nCouldn't find {csv_file}, creating some test data instead.")
        
        # Generate synthetic data with known pattern
        np.random.seed(42)
        n_samples, n_features = 50, 3
        X = np.random.randn(n_samples, n_features)
        true_coef = np.array([1.5, 0.0, -2.0])
        y = X @ true_coef + np.random.randn(n_samples) * 0.5
        
        # Save for future test runs
        try:
            with open(csv_file, "w") as f:
                f.write("x_0,x_1,x_2,y\n")
                for i in range(n_samples):
                    f.write(f"{X[i,0]},{X[i,1]},{X[i,2]},{y[i]}\n")
            print(f"Saved test data to {csv_file}")
        except:
            print("Couldn't save the test data, but we'll use it anyway")

    # Run OLS (alpha=0) to test basic functionality
    model = LassoHomotopyModel(alpha=0.0, max_iter=5000, tol=1e-5)
    results = model.fit(X, y)

    # Get predictions
    preds = results.predict(X)

    # Show some results
    print("\n=== Model Test Results ===")
    print("Coefficients:", results.coef_)
    print("Intercept:", results.intercept_)
    for i in range(min(5, len(X))):
        print(f"Example {i}: Predicted={preds[i]:.4f}, Actual={y[i]:.4f}")

    # Calculate error
    mse = np.mean((preds - y)**2)
    print(f"\nMSE: {mse:.4f}")

    # Should be able to fit the data well
    assert mse < threshold, f"Error too high! MSE={mse:.4f}, threshold={threshold}"

def test_simple_case_noncollinear():
    """
    Tests the model with a simple, clean dataset.
    
    Uses data that follows a clear pattern and checks both OLS and LASSO.
    """
    # Simple data that roughly follows y = 2*x_0 + 1*x_1 + 1
    X = np.array([
        [1, 1],
        [2, 1],
        [3, 2],
        [4, 3],
        [5, 2]
    ])
    y = np.array([4, 6, 9, 12, 13])
    
    # First try with OLS
    model = LassoHomotopyModel(alpha=0.0)
    results = model.fit(X, y)
    
    # Check fit quality
    preds = results.predict(X)
    mse = np.mean((preds - y)**2)
    print(f"Simple case MSE: {mse:.4f}")
    assert mse < 0.1, f"OLS should fit this data almost perfectly, but got MSE={mse}"
    
    # Now try with LASSO regularization
    model_lasso = LassoHomotopyModel(alpha=0.1)
    results_lasso = model_lasso.fit(X, y)
    
    # Check LASSO predictions
    preds_lasso = results_lasso.predict(X)
    mse_lasso = np.mean((preds_lasso - y)**2)
    print(f"LASSO MSE with alpha=0.1: {mse_lasso:.4f}")
    
    # Just make sure we get valid results
    assert np.isfinite(mse_lasso), "LASSO predictions should be valid numbers"
    
    # Try with smaller regularization
    model_tiny_alpha = LassoHomotopyModel(alpha=0.01)
    results_tiny_alpha = model_tiny_alpha.fit(X, y)
    preds_tiny_alpha = results_tiny_alpha.predict(X)
    mse_tiny_alpha = np.mean((preds_tiny_alpha - y)**2)
    print(f"LASSO MSE with alpha=0.01: {mse_tiny_alpha:.4f}")
    
    # Show how coefficients change with regularization
    print("OLS coefficients:", results.coef_)
    print("LASSO (alpha=0.1) coefficients:", results_lasso.coef_)
    print("LASSO (alpha=0.01) coefficients:", results_tiny_alpha.coef_)

def test_ols_comparison_relaxed():
    """
    Makes sure our OLS implementation matches direct OLS solution.
    
    When alpha=0, our model should give essentially the same answer as 
    directly solving the normal equations.
    """
    # Create random test data
    np.random.seed(42)
    n_samples, n_features = 50, 4
    X = np.random.randn(n_samples, n_features)
    true_coef = np.array([1.5, -0.5, 0.8, 0])
    y = X @ true_coef + np.random.randn(n_samples) * 0.2
    
    # Fit with our model's OLS mode
    model = LassoHomotopyModel(alpha=0.0)
    results = model.fit(X, y)
    
    # Calculate direct OLS solution for comparison
    XTX = X.T @ X
    XTy = X.T @ y
    direct_coef = np.linalg.solve(XTX, XTy)
    
    # Let's see both results
    print("Our model coefficients:", results.coef_)
    print("Direct OLS coefficients:", direct_coef)
    
    # They should be very close
    max_diff = np.max(np.abs(results.coef_ - direct_coef))
    assert max_diff < 0.1, f"Coefficients differ too much, max difference = {max_diff}"
    
    # Check predictions too
    our_preds = results.predict(X)
    direct_preds = X @ direct_coef + results.intercept_
    pred_mse = np.mean((our_preds - direct_preds)**2)
    assert pred_mse < 0.01, f"Predictions differ too much, MSE = {pred_mse}"

def test_sparsity_pattern():
    """
    Tests if LASSO correctly identifies important features.
    
    Creates data where only some features matter and checks if
    LASSO picks them out.
    """
    # Create data where only the first 3 features are important
    np.random.seed(42)
    n_samples, n_features = 100, 10
    
    X = np.random.randn(n_samples, n_features)
    true_coef = np.zeros(n_features)
    true_coef[:3] = [3.0, -2.0, 1.0]  # Only first 3 features matter
    
    y = X @ true_coef + np.random.randn(n_samples) * 0.5
    
    # Try with small regularization
    model_small_alpha = LassoHomotopyModel(alpha=0.1)
    results_small = model_small_alpha.fit(X, y)
    
    # Then with stronger regularization
    model_large_alpha = LassoHomotopyModel(alpha=1.0)
    results_large = model_large_alpha.fit(X, y)
    
    # Count features that remain non-zero
    nonzero_small = np.sum(np.abs(results_small.coef_) > 1e-10)
    nonzero_large = np.sum(np.abs(results_large.coef_) > 1e-10)
    
    print(f"\nNon-zero coefficients with weak regularization: {nonzero_small}")
    print(f"Non-zero coefficients with strong regularization: {nonzero_large}")
    
    # Should find at least something
    assert nonzero_small > 0, "Model should find some important features, even with regularization"
    
    # Check if it found any of the truly important features
    important_features_used = np.sum(np.abs(results_small.coef_[:3]) > 1e-10)
    assert important_features_used > 0, "Model missed all the important features!"

def test_collinear_features():
    """
    Tests how LASSO handles correlated features.
    
    One advantage of LASSO is feature selection with correlated inputs.
    This test checks if it chooses a subset of correlated features.
    """
    # Make correlated features - this is where LASSO shines
    np.random.seed(42)
    n_samples = 100
    
    # Base feature
    x1 = np.random.randn(n_samples)
    
    # Two features highly correlated with x1
    x2 = x1 * 0.95 + np.random.randn(n_samples) * 0.1
    x3 = x1 * 0.9 + np.random.randn(n_samples) * 0.15
    
    # Unrelated feature
    x4 = np.random.randn(n_samples)
    
    # Put it all together
    X = np.column_stack([x1, x2, x3, x4])
    
    # Target only depends on x1 and x4
    y = 2*x1 + 0*x2 + 0*x3 + 1*x4 + np.random.randn(n_samples) * 0.2
    
    # Check correlations
    corr_matrix = np.corrcoef(X.T)
    print("\nFeature correlations:")
    print(corr_matrix)
    
    # Run OLS first
    ols_model = LassoHomotopyModel(alpha=0.0)
    ols_results = ols_model.fit(X, y)
    
    # Then LASSO
    lasso_model = LassoHomotopyModel(alpha=0.5)
    lasso_results = lasso_model.fit(X, y)
    
    # Show coefficients
    print("\nOLS found:", ols_results.coef_)
    print("LASSO found:", lasso_results.coef_)
    
    # Count non-zeros
    ols_nonzero = np.sum(np.abs(ols_results.coef_) > 1e-10)
    lasso_nonzero = np.sum(np.abs(lasso_results.coef_) > 1e-10)
    
    print(f"OLS used {ols_nonzero} features")
    print(f"LASSO used {lasso_nonzero} features")
    print(f"LASSO was {ols_nonzero - lasso_nonzero} features more sparse than OLS")

def test_increasing_alpha():
    """
    Tests how regularization strength affects sparsity.
    
    As alpha increases, more coefficients should be set to zero.
    """
    # Create data with known pattern
    np.random.seed(42)
    n_samples, n_features = 100, 8
    X = np.random.randn(n_samples, n_features)
    true_coef = np.array([3, 1.5, 0, 0, 2, 0, 0, 0.5])  # Some zeros
    y = X @ true_coef + np.random.randn(n_samples) * 0.3
    
    # Try several regularization strengths
    alphas = [0, 0.01, 0.1, 0.5, 1.0, 2.0]  # From none to strong
    nonzeros = []
    
    for alpha in alphas:
        model = LassoHomotopyModel(alpha=alpha)
        results = model.fit(X, y)
        nonzero_count = np.sum(np.abs(results.coef_) > 1e-10)
        nonzeros.append(nonzero_count)
    
    print("\nRegularization strength:", alphas)
    print("Features used:", nonzeros)
    
    # Generally, stronger regularization should mean fewer features
    high_alpha_sparsity = np.mean(nonzeros[-2:])  # Average of highest alphas
    low_alpha_sparsity = np.mean(nonzeros[:2])    # Average of lowest alphas
    
    assert high_alpha_sparsity <= low_alpha_sparsity, \
        "Higher regularization should generally lead to fewer features"

def test_max_iter_effect():
    """
    Tests how iteration limit affects solution quality.
    
    More iterations typically means better convergence.
    """
    # Create moderately complex data
    np.random.seed(42)
    n_samples, n_features = 100, 20
    X = np.random.randn(n_samples, n_features)
    true_coef = np.zeros(n_features)
    true_coef[[0, 5, 10, 15]] = [2, -1, 1.5, -0.8]  # Sparse pattern
    y = X @ true_coef + np.random.randn(n_samples) * 0.2
    
    # Try with few, medium, and many iterations
    low_iter_model = LassoHomotopyModel(alpha=0.1, max_iter=3)
    medium_iter_model = LassoHomotopyModel(alpha=0.1, max_iter=10)
    high_iter_model = LassoHomotopyModel(alpha=0.1, max_iter=100)
    
    low_iter_results = low_iter_model.fit(X, y)
    medium_iter_results = medium_iter_model.fit(X, y)
    high_iter_results = high_iter_model.fit(X, y)
    
    # Check fit quality
    low_iter_mse = np.mean((low_iter_results.predict(X) - y)**2)
    medium_iter_mse = np.mean((medium_iter_results.predict(X) - y)**2)
    high_iter_mse = np.mean((high_iter_results.predict(X) - y)**2)
    
    print(f"\nError with 3 iterations: {low_iter_mse:.6f}")
    print(f"Error with 10 iterations: {medium_iter_mse:.6f}")
    print(f"Error with 100 iterations: {high_iter_mse:.6f}")
    
    # Check feature selection
    low_iter_nonzero = np.sum(np.abs(low_iter_results.coef_) > 1e-10)
    medium_iter_nonzero = np.sum(np.abs(medium_iter_results.coef_) > 1e-10)
    high_iter_nonzero = np.sum(np.abs(high_iter_results.coef_) > 1e-10)
    
    print(f"Features used with 3 iterations: {low_iter_nonzero}")
    print(f"Features used with 10 iterations: {medium_iter_nonzero}")
    print(f"Features used with 100 iterations: {high_iter_nonzero}")

def test_tolerance_effect():
    """
    Tests how convergence tolerance affects solution.
    
    Lower tolerance means more precise solutions but may take longer.
    """
    # Create data with specific pattern
    np.random.seed(42)
    n_samples, n_features = 100, 10
    X = np.random.randn(n_samples, n_features)
    true_coef = np.array([2, 0, 1.5, 0, 0, -1, 0, 0, 0.8, 0])  # Some zeros
    y = X @ true_coef + np.random.randn(n_samples) * 0.1
    
    # Try different tolerance levels
    high_tol_model = LassoHomotopyModel(alpha=0.1, tol=1e-2)  # Less precise
    medium_tol_model = LassoHomotopyModel(alpha=0.1, tol=1e-4)  # Medium
    low_tol_model = LassoHomotopyModel(alpha=0.1, tol=1e-6)  # More precise
    
    high_tol_results = high_tol_model.fit(X, y)
    medium_tol_results = medium_tol_model.fit(X, y)
    low_tol_results = low_tol_model.fit(X, y)
    
    # Check error with each
    high_tol_mse = np.mean((high_tol_results.predict(X) - y)**2)
    medium_tol_mse = np.mean((medium_tol_results.predict(X) - y)**2)
    low_tol_mse = np.mean((low_tol_results.predict(X) - y)**2)
    
    print(f"\nError with loose tolerance: {high_tol_mse:.6f}")
    print(f"Error with medium tolerance: {medium_tol_mse:.6f}")
    print(f"Error with tight tolerance: {low_tol_mse:.6f}")

@pytest.mark.skipif(not sklearn_available, reason="scikit-learn not installed")
def test_against_sklearn_lasso_relaxed():
    """
    Compares our LASSO with scikit-learn's implementation.
    
    They should be somewhat similar, though exact details may differ.
    Skips this test if scikit-learn isn't installed.
    """
    # Generate regression data with sklearn
    np.random.seed(42)
    n_samples, n_features = 100, 15
    X, y, true_coef = make_regression(
        n_samples=n_samples, 
        n_features=n_features,
        n_informative=5,  # Only 5 features matter
        coef=True, 
        random_state=42,
        noise=5
    )
    
    # Scale for better numerics
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Implementations may scale alpha differently
    our_alpha = 0.5
    sklearn_alpha = 0.1
    
    # Fit both models
    our_model = LassoHomotopyModel(alpha=our_alpha)
    our_results = our_model.fit(X_scaled, y)
    
    sklearn_model = SklearnLasso(alpha=sklearn_alpha, fit_intercept=True)
    sklearn_model.fit(X_scaled, y)
    
    # Compare predictions
    our_preds = our_results.predict(X_scaled)
    sklearn_preds = sklearn_model.predict(X_scaled)
    
    our_mse = np.mean((our_preds - y)**2)
    sklearn_mse = np.mean((sklearn_preds - y)**2)
    
    print("\nOur model error:", our_mse)
    print("Scikit-learn error:", sklearn_mse)
    
    # Check prediction correlation
    prediction_corr = np.corrcoef(our_preds, sklearn_preds)[0, 1]
    print(f"Prediction correlation: {prediction_corr:.6f}")
    
    # Should be reasonably correlated
    assert prediction_corr > 0.6, "Our predictions should correlate with scikit-learn's"
    
    # Compare sparsity
    our_nonzero = np.sum(np.abs(our_results.coef_) > 1e-10)
    sklearn_nonzero = np.sum(np.abs(sklearn_model.coef_) > 1e-10)
    
    print(f"Our model used {our_nonzero} features")
    print(f"Scikit-learn used {sklearn_nonzero} features")
    print(f"Difference: {abs(our_nonzero - sklearn_nonzero)} features")

def test_singular_matrix_handling_relaxed():
    """
    Tests model on nearly singular matrices.
    
    This is a tough case where OLS can fail, but LASSO should work.
    """
    # Create challenging data (nearly rank-deficient)
    np.random.seed(42)
    n_samples, n_features = 30, 25  # Close to singular
    X = np.random.randn(n_samples, n_features)
    
    # Only two features actually matter
    true_coef = np.zeros(n_features)
    true_coef[[5, 15]] = [1, -1]
    y = X @ true_coef + np.random.randn(n_samples) * 0.2
    
    # LASSO should still work
    model = LassoHomotopyModel(alpha=0.1)
    results = model.fit(X, y)
    
    # It should produce something meaningful
    assert results is not None, "Model failed to run"
    assert results.coef_ is not None, "No coefficients produced"
    
    # Check sparsity and fit
    nonzero_count = np.sum(np.abs(results.coef_) > 1e-10)
    print(f"\nFeatures used on challenging problem: {nonzero_count}")
    
    preds = results.predict(X)
    mse = np.mean((preds - y)**2)
    print(f"Error on challenging problem: {mse:.6f}")
    
    # Should give valid predictions
    assert np.isfinite(mse), "Error should be a valid number"

def test_perfect_collinearity():
    """
    Tests with perfectly correlated features.
    
    One key advantage of LASSO is handling perfect collinearity by
    selecting just one from a set of identical features.
    """
    # Create data with perfect collinearity
    np.random.seed(42)
    n_samples = 100
    x1 = np.random.randn(n_samples)
    x2 = 2 * x1  # Exactly collinear with x1
    x3 = np.random.randn(n_samples)  # Independent feature
    
    X = np.column_stack([x1, x2, x3])
    
    # True relationship only uses x1 and x3
    y = 3 * x1 + 0 * x2 + 2 * x3 + np.random.randn(n_samples) * 0.1
    
    # Run LASSO
    model = LassoHomotopyModel(alpha=0.5)
    results = model.fit(X, y)
    
    # Show what it found
    print("Coefficients with perfect collinearity:", results.coef_)
    
    # It should pick at most one from the collinear pair
    both_nonzero = (abs(results.coef_[0]) > 1e-10) and (abs(results.coef_[1]) > 1e-10)
    assert not both_nonzero, "LASSO should pick at most one from perfectly collinear features"

def test_group_collinearity():
    """
    Tests with groups of related features.
    
    Real data often has groups of related variables. LASSO should
    select representatives from each important group.
    """
    # Create groups of correlated features
    np.random.seed(42)
    n_samples = 200
    
    # First correlated group
    x1 = np.random.randn(n_samples)
    x2 = x1 + np.random.randn(n_samples) * 0.05  # Similar to x1
    x3 = x1 + np.random.randn(n_samples) * 0.08  # Similar to x1
    
    # Second correlated group
    x4 = np.random.randn(n_samples)
    x5 = x4 + np.random.randn(n_samples) * 0.1  # Similar to x4
    x6 = x4 + np.random.randn(n_samples) * 0.12  # Similar to x4
    
    # Independent features
    x7 = np.random.randn(n_samples)
    x8 = np.random.randn(n_samples)
    
    X = np.column_stack([x1, x2, x3, x4, x5, x6, x7, x8])
    
    # Only need one from each group plus one independent
    y = 2*x1 + 0*x2 + 0*x3 + 1.5*x4 + 0*x5 + 0*x6 + x7 + 0*x8 + np.random.randn(n_samples) * 0.2
    
    # Try different regularization strengths
    alphas = [0.01, 0.1, 0.5, 1.0]  # Weak to strong
    for alpha in alphas:
        model = LassoHomotopyModel(alpha=alpha)
        results = model.fit(X, y)
        
        # Count features used in each group
        group1_nonzeros = sum(abs(results.coef_[i]) > 1e-10 for i in range(3))
        group2_nonzeros = sum(abs(results.coef_[i+3]) > 1e-10 for i in range(3))
        
        print(f"Alpha={alpha}: Group1 uses {group1_nonzeros} features, Group2 uses {group2_nonzeros} features")
        
        # With enough regularization, should use only 1-2 per group
        if alpha >= 0.1:
            assert group1_nonzeros <= 2, f"Too many features from group 1: {group1_nonzeros}"
            assert group2_nonzeros <= 2, f"Too many features from group 2: {group2_nonzeros}"

def test_ill_conditioned_data():
    """
    Tests with numerically challenging data.
    
    Creates a matrix with very high condition number to test stability.
    """
    # Create ill-conditioned matrix
    np.random.seed(42)
    n_samples, n_features = 100, 10
    
    # Start with random data
    X_base = np.random.randn(n_samples, n_features)
    
    # Break it down
    U, s, Vt = np.linalg.svd(X_base, full_matrices=False)
    
    # Make it ill-conditioned by shrinking some singular values
    s[1:] = s[1:] / 1000
    
    # Put it back together
    X = U @ np.diag(s) @ Vt
    
    # Create target with known coefficients
    true_coef = np.zeros(n_features)
    true_coef[0] = 1.0
    true_coef[5] = 0.5
    
    y = X @ true_coef + np.random.randn(n_samples) * 0.01
    
    # Try OLS first
    try:
        XTX = X.T @ X
        XTX_inv = np.linalg.inv(XTX)
        ols_coef = XTX_inv @ (X.T @ y)
        ols_stable = True
    except np.linalg.LinAlgError:
        ols_stable = False
    
    # Now LASSO
    model = LassoHomotopyModel(alpha=0.1)
    results = model.fit(X, y)
    
    # Check how ill-conditioned our matrix is
    cond_num = np.linalg.cond(X)
    print(f"Matrix condition number: {cond_num:.2f}")
    
    # Show LASSO's solution
    print("LASSO coefficients:", results.coef_)
    
    # Should at least give valid results
    assert np.isfinite(results.coef_).all(), "Coefficients should be valid numbers"
    assert np.isfinite(results.predict(X)).all(), "Predictions should be valid numbers"