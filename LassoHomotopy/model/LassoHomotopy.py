import numpy as np

class LassoHomotopyModel:
    def __init__(self, alpha=1.0, max_iter=500, tol=1e-4):
        # Check for negative alpha
        if alpha < 0:
            raise ValueError("Alpha must be non-negative.")
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

        self.coef_ = None
        self.intercept_ = None

        print(f"DEBUG: alpha={alpha}, max_iter={max_iter}, tol={tol}")

    def fit(self, X, y):
        # Convert input to float arrays
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        # Check for empty datasets
        if X.size == 0 or y.size == 0:
            raise ValueError("Empty dataset provided.")
        # Check for NaN values
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("Input contains NaN values.")
        # Check dimension mismatch: number of samples must agree
        if X.shape[0] != y.shape[0]:
            raise ValueError("Mismatched number of samples between X and y.")
        
        # If alpha==0, use OLS directly.
        if self.alpha == 0.0:
            return self._fit_ols(X, y)
        else:
            return self._fit_homotopy(X, y)

    def _fit_ols(self, X, y):
        n_samples, n_features = X.shape

        # 1) Center data
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        Xc = X - X_mean
        yc = y - y_mean

        # 2) Solve normal equations
        Gram = Xc.T @ Xc
        try:
            Gram_inv = np.linalg.inv(Gram)
        except np.linalg.LinAlgError:
            raise ValueError("Singular matrix encountered in OLS.")
        coef = Gram_inv @ (Xc.T @ yc)

        # 3) Uncenter: intercept = y_mean - X_mean.dot(coef)
        self.coef_ = coef
        self.intercept_ = y_mean - X_mean.dot(coef)

        result = LassoHomotopyResults(self.coef_, self.intercept_)
        # For OLS, we can record a single error value if desired.
        result.errors = [np.mean((yc - Xc @ coef) ** 2)]
        return result

    def _fit_homotopy(self, X, y):
        n_samples, n_features = X.shape

        # Center data
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        Xc = X - X_mean
        yc = y - y_mean

        # Initialize coefficients and intercept
        self.coef_ = np.zeros(n_features, dtype=float)
        self.intercept_ = y_mean

        # Active set of features and their signs
        active_idx = []
        signs = np.zeros(n_features, dtype=float)

        # Initialize residual and correlations
        residual = yc.copy()
        corr = Xc.T @ residual

        # Initialize list to store error progression for convergence logging
        errors = []
        errors.append(np.mean(residual**2))

        for iteration in range(self.max_iter):
            print(f"Iteration={iteration} | active_idx={active_idx} | coef_={self.coef_}")
            c_abs = np.abs(corr)
            c_max = np.max(c_abs)
            
            # Stop if maximum correlation falls below alpha
            if c_max < self.alpha:
                break

            # Add feature with largest correlation if not active
            j = np.argmax(c_abs)
            if j not in active_idx:
                active_idx.append(j)
                signs[j] = np.sign(corr[j])

            # Form submatrix for active features (adjusted by sign)
            Xa = Xc[:, active_idx] * signs[active_idx]

            # Compute Gram matrix for active set and invert
            G = Xa.T @ Xa
            try:
                G_inv = np.linalg.inv(G)
            except np.linalg.LinAlgError:
                break

            ones = np.ones(len(active_idx))
            A_factor = 1.0 / np.sqrt(ones @ G_inv @ ones)
            w = A_factor * (G_inv @ ones)  # direction for coefficients

            # Compute direction in residual space
            u = Xa @ w

            # Compute step size gamma for new events (entering or sign change)
            a_vec = Xc.T @ u
            gamma_candidates = []
            for k in range(n_features):
                if k not in active_idx:
                    ck = corr[k]
                    ak = a_vec[k]
                    if A_factor - ak != 0:
                        g1 = (c_max - ck) / (A_factor - ak)
                        if g1 > 0:
                            gamma_candidates.append(g1)
                    if A_factor + ak != 0:
                        g2 = (c_max + ck) / (A_factor + ak)
                        if g2 > 0:
                            gamma_candidates.append(g2)

            if not gamma_candidates:
                break

            gamma = min(gamma_candidates)

            # Update coefficients for active features
            for i, idx in enumerate(active_idx):
                self.coef_[idx] += gamma * signs[idx] * w[i]

            # Update residual and correlations
            residual -= gamma * u
            corr = Xc.T @ residual

            # Record error for convergence monitoring
            errors.append(np.mean(residual**2))

            # Check if any active coefficient's sign has flipped, remove it if so.
            to_remove = []
            for idx in active_idx:
                if np.sign(self.coef_[idx]) != signs[idx]:
                    self.coef_[idx] = 0.0
                    to_remove.append(idx)
            for idx in to_remove:
                active_idx.remove(idx)

            if np.abs(gamma) < self.tol:
                break

        result = LassoHomotopyResults(self.coef_, self.intercept_)
        result.errors = errors  # store error history for testing convergence
        return result


class LassoHomotopyResults:
    def __init__(self, coef, intercept):
        self.coef_ = coef
        self.intercept_ = intercept

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_
