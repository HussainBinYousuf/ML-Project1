import numpy as np

class LassoHomotopyModel:
    def __init__(self, alpha=1.0, max_iter=500, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

        self.coef_ = None
        self.intercept_ = None

        print(f"DEBUG: alpha={alpha}, max_iter={max_iter}, tol={tol}")

    def fit(self, X, y):
        """
        If alpha == 0, we solve ordinary least squares (OLS) directly.
        Otherwise, we use the Homotopy/LARS steps for LASSO.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        # If alpha=0, do direct OLS instead of Homotopy logic.
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

        # 2) Solve
        Gram = Xc.T @ Xc
        Gram_inv = np.linalg.inv(Gram)
        coef = Gram_inv @ (Xc.T @ yc)

        # 3) Uncenter properly
        self.coef_ = coef
        self.intercept_ = y_mean - X_mean.dot(coef)

        return LassoHomotopyResults(self.coef_, self.intercept_)


    def _fit_homotopy(self, X, y):
        """
        Use the Homotopy (LARS-like) approach for alpha > 0.
        """
        n_samples, n_features = X.shape

        # Center data to simplify computation (handle intercept separately)
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        Xc = X - X_mean
        yc = y - y_mean

        # Initialize coefficients to zeros; intercept will be y_mean.
        self.coef_ = np.zeros(n_features, dtype=float)
        self.intercept_ = y_mean

        # Active set of features and their signs.
        active_idx = []
        signs = np.zeros(n_features, dtype=float)

        # Initial residual and correlations.
        residual = yc.copy()
        corr = Xc.T @ residual

        for iteration in range(self.max_iter):
            print(f"Iteration={iteration} | active_idx={active_idx} | coef_={self.coef_}")
            c_abs = np.abs(corr)
            c_max = np.max(c_abs)
            
            # Stop if maximum correlation falls below alpha
            if c_max < self.alpha:
                break

            # Add the feature with the largest correlation if not active
            j = np.argmax(c_abs)
            if j not in active_idx:
                active_idx.append(j)
                signs[j] = np.sign(corr[j])

            # Form submatrix with active features (adjusted by sign)
            Xa = Xc[:, active_idx] * signs[active_idx]

            # Compute the Gram matrix and its inverse for the active set
            G = Xa.T @ Xa
            try:
                G_inv = np.linalg.inv(G)
            except np.linalg.LinAlgError:
                # In case the matrix is singular, break out of the loop.
                break

            ones = np.ones(len(active_idx))
            A_factor = 1.0 / np.sqrt(ones @ G_inv @ ones)
            w = A_factor * (G_inv @ ones)  # direction for coefficients

            # Compute the direction in residual space
            u = Xa @ w

            # Compute step size gamma for new events (entering or sign change)
            a_vec = Xc.T @ u
            gamma_candidates = []
            for k in range(n_features):
                if k not in active_idx:
                    ck = corr[k]
                    ak = a_vec[k]
                    # Two potential ways to hit a boundary (avoid /0)
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

            # Update coefficients for the active features.
            for i, idx in enumerate(active_idx):
                self.coef_[idx] += gamma * signs[idx] * w[i]

            # Update residual and correlations.
            residual -= gamma * u
            corr = Xc.T @ residual

            # Check for coefficients that cross zero and remove them.
            to_remove = []
            for idx in active_idx:
                # If a coefficient sign flips, set it to 0 and remove
                if np.sign(self.coef_[idx]) != signs[idx]:
                    self.coef_[idx] = 0.0
                    to_remove.append(idx)

            for idx in to_remove:
                active_idx.remove(idx)

            # Stop if gamma is too small
            if np.abs(gamma) < self.tol:
                break

        return LassoHomotopyResults(self.coef_, self.intercept_)


class LassoHomotopyResults:
    def __init__(self, coef, intercept):
        self.coef_ = coef
        self.intercept_ = intercept

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_
