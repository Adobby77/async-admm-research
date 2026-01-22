import numpy as np

class ADMMLasso:
    def __init__(self, lambda_reg=1.0, rho=1.0, max_iter=1000, tol=1e-4):
        """
        ADMM solver for the Lasso problem:
        min_x 0.5 * ||Ax - b||_2^2 + lambda_reg * ||x||_1

        Args:
            lambda_reg (float): L1 regularization parameter.
            rho (float): ADMM penalty parameter.
            max_iter (int): Maximum number of iterations.
            tol (float): Tolerance for convergence (primal and dual residuals).
        """
        self.lambda_reg = lambda_reg
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, A, b):
        """
        Solve the Lasso problem using ADMM.

        Args:
            A (np.ndarray): Design matrix of shape (m, n).
            b (np.ndarray): Target vector of shape (m,).

        Returns:
            np.ndarray: Solution vector x of shape (n,).
            dict: History of objective values and residuals.
        """
        m, n = A.shape
        
        # Initialize variables
        x = np.zeros(n)
        z = np.zeros(n)
        u = np.zeros(n)

        # Precompute Cholesky decomposition for x-update
        # (A^T A + rho * I) x = A^T b + rho(z - u)
        Atb = A.T @ b
        AtA = A.T @ A
        # Use Cholesky: L L^T = A^T A + rho I
        L = np.linalg.cholesky(AtA + self.rho * np.eye(n))
        L_inv = np.linalg.inv(L) # Not strictly necessary to invert, can use solve, but okay for this
        # Or better, use scipy.linalg.solve_triangular if available, but pure numpy:
        # We will use np.linalg.solve twice in the loop, or pre-invert (AtA + rho I) if n is small
        # For this implementation, let's stick to using the cholesky factor for solves.

        history = {'obj': [], 'r_norm': [], 's_norm': []}

        for k in range(self.max_iter):
            # --- x-update ---
            # (A^T A + rho I) x = A^T b + rho(z - u)
            q = Atb + self.rho * (z - u)
            # Forward solve L y = q
            # Backward solve L^T x = y
            # Using low-level linalg solve
            # x = (A^T A + rho I)^-1 q
            # Using Cholesky:
            y_sol = np.linalg.solve(L, q)
            x = np.linalg.solve(L.T, y_sol)

            # --- z-update ---
            # z = SoftThreshold(x + u, lambda / rho)
            z_old = z.copy()
            z = self._soft_threshold(x + u, self.lambda_reg / self.rho)

            # --- u-update ---
            u = u + x - z

            # --- Convergence check ---
            # Primal residual: r = x - z
            r = x - z
            # Dual residual: s = rho * (z - z_old)
            s = self.rho * (z - z_old)

            r_norm = np.linalg.norm(r)
            s_norm = np.linalg.norm(s)

            obj = 0.5 * np.linalg.norm(A @ x - b)**2 + self.lambda_reg * np.linalg.norm(x, 1)
            
            history['obj'].append(obj)
            history['r_norm'].append(r_norm)
            history['s_norm'].append(s_norm)

            if r_norm < self.tol and s_norm < self.tol:
                print(f"Converged at iteration {k+1}")
                break
        
        return x, history

    def _soft_threshold(self, v, kappa):
        """
        Soft thresholding operator S_kappa(v).
        """
        return np.sign(v) * np.maximum(np.abs(v) - kappa, 0.0)
