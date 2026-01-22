import numpy as np
import matplotlib.pyplot as plt
import time
from admm_lasso import ADMMLasso

def main():
    # 1. Generate synthetic data
    np.random.seed(42)
    n_samples, n_features = 100, 200 # Over-determined or under-determined
    # Let's try under-determined (n_features > n_samples) typical for Lasso compressed sensing
    n_samples, n_features = 50, 100 
    
    # Random design matrix
    A = np.random.randn(n_samples, n_features)
    
    # Sparse true signal
    n_active = 5
    true_x = np.zeros(n_features)
    active_indices = np.random.choice(n_features, n_active, replace=False)
    true_x[active_indices] = np.random.randn(n_active)
    
    # Observations with noise
    sigma = 0.1
    b = A @ true_x + sigma * np.random.randn(n_samples)

    lambda_reg = 1.0
    rho = 1.0

    print(f"Problem dimensions: A: {A.shape}, b: {b.shape}")
    print(f"Simulating sparse signal with {n_active} non-zero components.")
    
    # 2. Run ADMM Lasso
    print("\n--- Running ADMM Lasso ---")
    admm = ADMMLasso(lambda_reg=lambda_reg, rho=rho, max_iter=1000, tol=1e-4)
    start_time = time.time()
    x_admm, history = admm.fit(A, b)
    end_time = time.time()
    
    print(f"ADMM finished in {end_time - start_time:.4f} seconds.")
    print(f"Final objective: {history['obj'][-1]:.4f}")
    
    # 3. Compare with Truth
    print("\n--- Results Analysis ---")
    mse = np.mean((x_admm - true_x)**2)
    print(f"MSE vs True Signal: {mse:.6f}")
    
    print("Top 10 estimated coefficients by magnitude:")
    top_indices = np.argsort(np.abs(x_admm))[::-1][:10]
    for idx in top_indices:
        print(f"Index {idx:3d}: True={true_x[idx]:.4f}, Est={x_admm[idx]:.4f}")

    # 4. Compare with Scikit-Learn (if available)
    try:
        from sklearn.linear_model import Lasso
        print("\n--- Comparing with Scikit-Learn Lasso ---")
        # sklearn Lasso objective is (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
        # Our objective is 0.5 * ||Ax - b||_2^2 + lambda * ||x||_1
        # To match, we need to adjust alpha.
        # factor * (1/2n) * ||..||^2 + alpha ||..||_1  == 0.5 * ||..||^2 + lambda ||..||_1
        # Multiply sklearn obj by n_samples: 0.5 * ||..||^2 + (alpha * n_samples) ||..||_1
        # So effective lambda for us corresponds to alpha * n_samples in sklearn terms
        # So alpha = lambda / n_samples
        
        alpha_sklearn = lambda_reg / n_samples
        sklearn_lasso = Lasso(alpha=alpha_sklearn, fit_intercept=False, tol=1e-4, max_iter=2000)
        sklearn_lasso.fit(A, b)
        
        x_sklearn = sklearn_lasso.coef_
        mse_sklearn = np.mean((x_sklearn - true_x)**2)
        mse_vs_sklearn = np.mean((x_admm - x_sklearn)**2)
        
        print(f"Sklearn MSE vs True Signal: {mse_sklearn:.6f}")
        print(f"MSE ADMM vs Sklearn: {mse_vs_sklearn:.6f}")
        
    except ImportError:
        print("\nScikit-learn not found. Skipping comparison.")

    # 5. Visualization
    print("\n--- Generating Convergence Plot ---")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Residuals
    axs[0].plot(history['r_norm'], label='Primal Residual (r)')
    axs[0].plot(history['s_norm'], label='Dual Residual (s)')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Norm')
    axs[0].set_title('Convergence of Residuals')
    axs[0].legend()
    axs[0].grid(True)

    # Plot 2: Objective Value
    axs[1].plot(history['obj'])
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Objective Value')
    axs[1].set_title('Objective Function')
    axs[1].grid(True)

    # Plot 3: Coefficients Comparison
    axs[2].stem(true_x, linefmt='g-', markerfmt='go', basefmt=' ', label='True')
    axs[2].stem(x_admm, linefmt='r--', markerfmt='rx', basefmt=' ', label='Estimated')
    # Make it readable if too many features, maybe just plot indices? 
    # But stem is okay for sparse signals.
    axs[2].set_title('Coefficients Recovery')
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig('convergence_plot.png')
    print("Plot saved to 'convergence_plot.png'")
    # plt.show() # Cannot show in this environment

if __name__ == "__main__":
    main()
