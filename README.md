# Group Members:- 
## Fnu Hussain Bin Yousuf 
## Pushkraj Kohok 
## Rohit Lahori 
## Rajni Pawar

# Project 1 

Your objective is to implement the LASSO regularized regression model using the Homotopy Method. You can read about this method in [this](https://people.eecs.berkeley.edu/~elghaoui/Pubs/hom_lasso_NIPS08.pdf) paper and the references therein. You are required to write a README for your project. Please describe how to run the code in your project *in your README*. Including some usage examples would be an excellent idea. You may use Numpy/Scipy, but you may not use built-in models from, e.g. SciKit Learn. This implementation must be done from first principles. You may use SciKit Learn as a source of test data.

You should create a virtual environment and install the packages in the requirements.txt in your virtual environment. You can read more about virtual environments [here](https://docs.python.org/3/library/venv.html). Once you've installed PyTest, you can run the `pytest` CLI command *from the tests* directory. I would encourage you to add your own tests as you go to ensure that your model is working as a LASSO model should (Hint: What should happen when you feed it highly collinear data?)

In order to turn your project in: Create a fork of this repository, fill out the relevant model classes with the correct logic. Please write a number of tests that ensure that your LASSO model is working correctly. It should produce a sparse solution in cases where there is collinear training data. You may check small test sets into GitHub, but if you have a larger one (over, say 20MB), please let us know and we will find an alternative solution. In order for us to consider your project, you *must* open a pull request on this repo. This is how we consider your project is "turned in" and thus we will use the datetime of your pull request as your submission time. If you fail to do this, we will grade your project as if it is late, and your grade will reflect this as detailed on the course syllabus. 

You may include Jupyter notebooks as visualizations or to help explain what your model does, but you will be graded on whether your model is correctly implemented in the model class files and whether we feel your test coverage is adequate. We may award bonus points for compelling improvements/explanations above and beyond the assignment.

Put your README here. Answer the following questions.

* What does the model you have implemented do and when should it be used?
* How did you test your model to determine if it is working reasonably correctly?
* What parameters have you exposed to users of your implementation in order to tune performance? 
* Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

# LASSO Homotopy Regression Model

## Overview

This project implements a LASSO (Least Absolute Shrinkage and Selection Operator) regression model using the Homotopy Method. The model is built from scratch-without relying on pre-built solvers (e.g., from scikit-learn) and is designed to efficiently solve the ℓ1-regularized least squares problem. The primary goals are to enable both prediction and feature selection, especially in settings where data is high-dimensional, sparse, or exhibits collinearity.

---

## Project Questions & Answers

### 1. What does the model you have implemented do and when should it be used?

**Answer:**  
The LASSO Homotopy model minimizes the least squares error with an ℓ1 penalty on the coefficients. This penalty enforces sparsity by driving many coefficients to zero, thereby performing feature selection.

- **Use Cases:**
  - **Feature Selection:** When only a few predictors are believed to be significant, LASSO helps by discarding irrelevant features.
  - **High-Dimensional Data:** Particularly effective when the number of features exceeds the number of samples.
  - **Collinear Data:** Handles multicollinearity by selecting a representative subset from highly correlated features.

- **When to Use:**
  - When interpretability and model sparsity are desired.
  - When dealing with noisy or redundant predictors, where traditional OLS might overfit.

---

### 2. How did you test your model to determine if it is working reasonably correctly?

**Answer:**  
A comprehensive test suite (run via `pytest`) validates the model under various scenarios:

- **Basic Functionality:**  
  - *test_simple_case_noncollinear()*: Fits a small, clean dataset.
  - *test_ols_comparison_relaxed()*: Compares the OLS (α = 0) mode with the direct OLS solution.

- **Sparsity and Collinearity:**  
  - *test_sparsity_pattern()*, *test_collinear_features()*, *test_perfect_collinearity()*, *test_group_collinearity()*: Ensure proper sparsity and handling of collinear inputs.

- **Parameter Sensitivity & Convergence:**  
  - *test_increasing_alpha()*, *test_max_iter_effect()*, *test_tolerance_effect()*: Assess how changes in regularization strength, maximum iterations, and tolerance affect performance.

- **Robustness:**  
  - *test_singular_matrix_handling_relaxed()* and *test_ill_conditioned_data()*: Confirm model stability on nearly singular or ill-conditioned data.
  - *test_against_sklearn_lasso_relaxed()*: (When scikit-learn is available) Compares predictions with scikit-learn's Lasso.

- **Edge and Combined Anomaly Cases:**  
  - Additional tests (totaling 25 unique cases) cover reproducibility, non-numeric inputs, no-feature scenarios, outlier sensitivity, and combinations such as high collinearity with missing values.

These tests ensure that the model performs correctly under typical conditions and various edge cases.

---

### 3. What parameters have you exposed to users of your implementation in order to tune performance?

**Answer:**  
The implementation exposes several parameters to help users fine-tune the model:

- **alpha:**  
  - Controls regularization strength; higher values promote sparsity.
- **max_iter:**  
  - Maximum number of iterations for the homotopy algorithm; increasing this value can help achieve convergence in challenging cases.
- **tol:**  
  - Convergence tolerance; lower values yield more precise solutions at the cost of increased computation.
- **seed:**  
  - Sets the random seed for reproducibility in data generation and model initialization.
- **Data Generation Parameters:**  
  - In the `generate_regression_data.py` script, parameters such as the number of samples (`N`), noise scale (`scale`), and weight range (`rnge`) allow customization of synthetic data.

These parameters offer users control over the balance between sparsity, accuracy, convergence speed, and computational cost.

---

### 4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

**Answer:**  
- **Troublesome Inputs:**
  - **Highly Collinear Data:**  
    While the model is designed to manage collinearity by selecting a representative feature, extremely high or perfect collinearity can lead to instability in the coefficient estimates.
  - **Ill-Conditioned Data:**  
    Data with a very high condition number may cause numerical precision issues affecting convergence.
  - **Non-Numeric or Malformed Inputs:**  
    The model expects numeric input; mismatched dimensions or non-numeric values will trigger errors.

- **Potential Workarounds:**
  - **Enhanced Preprocessing:**  
    Incorporate feature scaling, variance filtering, or dimensionality reduction to stabilize the inputs.
  - **Algorithmic Enhancements:**  
    Implement adaptive regularization parameters or more robust convergence criteria to handle extreme collinearity or ill-conditioning.
  - **Improved Input Validation:**  
    Add clearer error messages and preprocessing steps to manage non-numeric or malformed inputs.

- **Fundamental Limitations:**  
  Some issues, like perfect collinearity, are inherent to ℓ1-regularized methods. Although the model can only select one among perfectly collinear features, additional assumptions or preprocessing might be necessary to recover the exact true coefficients.

---

## How to Run the Project

### 1. Set Up the Environment

1. **Create and Activate a Virtual Environment**

   - Navigate to the project directory.
   - Create a virtual environment by running:
     
     ```
     python3 -m venv <virtual_env_name>
     ```

   - Activate the virtual environment:
     
     - On macOS/Linux:
       
       ```
       source <virtual_env_name>/bin/activate
       ```
     
     - On Windows:
       
       ```
       <virtual_env_name>\Scripts\activate
       ```

2. **Install Dependencies**

   - Install the required packages using the provided requirements file:
     
     ```
     pip install -r requirements.txt
     ```

---

### 2. Generate Regression Data

Use the provided data generation script to create a synthetic dataset. For example, run:
    ```
     python generate_regression_data.py \
    -N 100 \
    -m 2.0 3.0 \
    -b 5 \
    -scale 0.1 \
    -rnge 0 10 \
    -seed 42 \
    -output_file data.csv
     ```

**Parameter Details:**
- **-N 100:** Generate 100 samples.
- **-m 0 1 2 3 4 5 6 7 8 9:** Specifies indices for non-zero coefficients.
- **-b 5:** Total number of non-zero coefficients.
- **-scale 0.1:** Noise scaling factor.
- **-rnge 0 10:** Range for generating coefficient weights.
- **-seed 42:** initializes the random number generator to a fixed state so that every time we run our code, the same sequence of "random" numbers is produced.
- **-output_file regression_data.csv:** Saves the generated dataset to a CSV file.

---

### 3. Run the Model

There are two main methods to run the model:

#### a. Run the Test Suite

Validate the model's implementation by executing the comprehensive test suite:
    ```
     pytest LassoHomotopy/tests/test_LassoHomotopy.py -v
     ```
This command runs tests covering basic functionality, sparsity, collinearity, robustness, and various edge-case scenarios.

#### b. Use the Jupyter Notebook for Interactive Exploration

For interactive visualization and experimentation, launch the provided Jupyter Notebook:
    ```
     jupyter notebook LassoHomotopy/notebook/visualization.ipynb
     ```

---

### 4. Run All Tests

To run the complete set of tests (including additional edge and combined anomaly cases), simply execute:
    ```
     pytest -v test_LassoHomotopy.py
     ```


## List of all the TEST CASES

The project has been rigorously tested using a comprehensive suite of **35 unique test cases**. These tests ensure that the LASSO Homotopy model performs reliably under a wide range of conditions—from typical datasets to extreme edge cases.

### Original Test Cases (15 Total)

1. **`test_simple_case_noncollinear()`**  
   *Tests the model's performance on a small, clean dataset with non-collinear features.*

2. **`test_ols_comparison_relaxed()`**  
   *Compares the model’s OLS mode (α = 0) with the direct OLS solution.*

3. **`test_sparsity_pattern()`**  
   *Verifies that the model enforces sparsity when only a few features are significant.*

4. **`test_collinear_features()`**  
   *Evaluates model behavior on datasets with highly correlated features.*

5. **`test_increasing_alpha()`**  
   *Checks that increasing the regularization parameter (α) results in fewer nonzero coefficients.*

6. **`test_max_iter_effect()`**  
   *Assesses the impact of maximum iterations on convergence and accuracy.*

7. **`test_tolerance_effect()`**  
   *Examines how different convergence tolerance settings affect the solution.*

8. **`test_against_sklearn_lasso_relaxed()`**  
   *Compares predictions with those from scikit-learn's Lasso (if available).*

9. **`test_singular_matrix_handling_relaxed()`**  
   *Ensures robustness when the input matrix is nearly singular.*

10. **`test_perfect_collinearity()`**  
    *Checks that with perfectly collinear features, the model selects a single representative feature.*

11. **`test_group_collinearity()`**  
    *Verifies that from groups of correlated features, the model picks a representative subset.*

12. **`test_ill_conditioned_data()`**  
    *Evaluates performance on numerically ill-conditioned datasets.*

13. **`test_generated_data()`**  
    *Uses synthetic data generation to confirm that the model fits well.*

14. **`test_reproducibility()`**  
    *Confirms that running the model twice with the same random seed and data produces identical results.*

15. **`test_non_numeric_input()`**  
    *Ensures that non-numeric input data triggers appropriate errors.*

### Additional Unique Test Cases (20 Total)

16. **`test_no_feature_input()`**  
    *Tests model behavior when the input dataset has zero columns (no features).*

17. **`test_outlier_sensitivity()`**  
    *Validates that the model remains stable and produces finite predictions even with extreme outliers.*

18. **`test_api_compliance()`**  
    *Checks that the fitted model exposes expected attributes and methods (e.g., `coef_`, `intercept_`, and `predict()`).*

19. **`test_monotonic_convergence()`** *(optional)*  
    *Verifies that error metrics decrease steadily (or do not increase unexpectedly) over iterations.*

20. **`test_highly_collinear_with_missing()`**  
    *Combines extreme collinearity with missing values to test error handling or data cleaning.*

21. **`test_high_dimensional_sparse_with_outliers()`**  
    *Assesses model performance on high-dimensional, sparse data with injected outlier values.*

22. **`test_mixed_scale_constant_nearconstant()`**  
    *Tests data with mixed scales, including constant and near-constant columns, to verify proper handling.*

23. **`test_mismatched_nonnumeric_outliers()`**  
    *Combines non-numeric input with extreme outlier values to check robust error handling.*

24. **`test_convergence_under_stress()`**  
    *Uses an extremely ill-conditioned matrix with heavy regularization to stress-test convergence and stability.*

25. **`test_sequential_data_changing_dynamics()`**  
    *Simulates sequential data with a shift in underlying dynamics to ensure the model produces valid outputs.*

26. **`test_intercept_stability()`**  
    *Checks that the model computes a stable intercept across different runs.*

27. **`test_prediction_consistency()`**  
    *Ensures that repeated predictions on the same dataset yield consistent results.*

28. **`test_input_with_constant_rows()`**  
    *Tests model behavior when some rows in the input matrix are constant.*

29. **`test_all_zero_input()`**  
    *Verifies that an input matrix composed entirely of zeros produces a zero coefficient vector.*

30. **`test_large_scale_inputs()`**  
    *Evaluates model performance when feature values are very large in magnitude.*

31. **`test_small_scale_inputs()`**  
    *Evaluates model performance when feature values are very small in magnitude.*

32. **`test_random_noise_robustness()`**  
    *Tests the model's robustness to significant random noise in the dataset.*

33. **`test_memory_efficiency()`**  
    *Checks that the model does not consume excessive memory on large datasets.*

34. **`test_runtime_performance()`**  
    *Measures the time taken to converge on a moderately large dataset to ensure acceptable performance.*

35. **`test_error_message_clarity()`**  
    *Verifies that the model produces clear and informative error messages for invalid inputs.*

---
Each test case has been designed to validate a specific aspect of the LASSO Homotopy model, ensuring its robustness, reliability, and correctness under a wide variety of conditions.


---

## Future Improvements

- **Enhanced Preprocessing:**
  - Integrate feature scaling, normalization, and dimensionality reduction (e.g., PCA) to stabilize input data, particularly for ill-conditioned or high-dimensional datasets.
  - Implement robust input validation to catch non-numeric or malformed data early.

- **Robustness Enhancements:**
  - Explore adaptive regularization strategies to automatically adjust the regularization parameter during training.
  - Improve convergence criteria by logging error metrics per iteration and possibly using dynamic tolerance thresholds.

- **Extended Testing and Monitoring:**
  - Expand the test suite to cover additional edge cases, such as mixed-scale inputs and sequential data with changing dynamics.
  - Add detailed logging for convergence metrics to help diagnose and optimize model performance.

- **Usability Enhancements:**
  - Develop a command-line interface (CLI) to allow users to easily adjust model parameters and run experiments directly from the terminal.
  - Enhance documentation with more usage examples, detailed parameter explanations, and troubleshooting guidelines.

---

## Troubleshooting

- **Environment Activation:**
  - Ensure that your virtual environment is properly activated (your prompt should indicate `(venv)`).
  - If the wrong Python interpreter is used (e.g., system Python instead of the one in your venv), verify your PATH settings or recreate the virtual environment.

- **Python Command Issues:**
  - On macOS/Linux, use `python3` if your system defaults to an older Python version.

- **Dependency Problems:**
  - Verify that the command `pip install -r requirements.txt` completes without errors.
  - If a package fails to install, check your Python version and ensure you have any required system libraries or compilers.

- **Test Failures:**
  - Run tests with increased verbosity (`pytest -v`) to see detailed error messages and help with debugging.
  - Make sure your generated data file (`regression_data.csv`) exists and is formatted correctly.

- **Jupyter Notebook:**
  - If the notebook does not launch, ensure Jupyter is installed in your virtual environment (`pip install jupyter`) and try running the `jupyter notebook` command again.

---
