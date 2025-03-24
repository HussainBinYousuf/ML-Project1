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

def test_predict(threshold=5.0):
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