import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist
import os
from sklearn.metrics import log_loss

# Set backend for Matplotlib to avoid GUI warnings in Flask
plt.switch_backend('Agg')

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

def generate_ellipsoid_clusters(distance, n_samples=100, cluster_std=0.3):  # Reduced cluster_std
    np.random.seed(0)
    covariance_matrix = np.array([[cluster_std, cluster_std * 0.5], 
                                [cluster_std * 0.5, cluster_std]])
    
    # Generate first cluster centered at origin
    X1 = np.random.multivariate_normal(mean=[0, 0], cov=covariance_matrix, size=n_samples)
    y1 = np.zeros(n_samples)

    # Generate second cluster with specified distance
    X2 = np.random.multivariate_normal(mean=[distance, distance], cov=covariance_matrix, size=n_samples)
    y2 = np.ones(n_samples)

    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))

    return X, y

# Function to fit logistic regression and extract coefficients
def fit_logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    beta0 = model.intercept_[0]
    beta1, beta2 = model.coef_[0]
    return model, beta0, beta1, beta2

def do_experiments(start, end, step_num):
    shift_distances = np.linspace(start, end, step_num)
    beta0_list, beta1_list, beta2_list, slope_list, intercept_list, loss_list, margin_widths = [], [], [], [], [], [], []

    n_samples = step_num
    n_cols = 2
    n_rows = (n_samples + n_cols - 1) // n_cols
    plt.figure(figsize=(20, n_rows * 10))

    for i, distance in enumerate(shift_distances, 1):
        X, y = generate_ellipsoid_clusters(distance=distance)
        # Add padding to the plot boundaries
        x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
        y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2

        # Fit logistic regression and append parameters
        model, beta0, beta1, beta2 = fit_logistic_regression(X, y)
        beta0_list.append(beta0)
        beta1_list.append(beta1)
        beta2_list.append(beta2)
        slope = -beta1 / beta2
        intercept = -beta0 / beta2
        slope_list.append(slope)
        intercept_list.append(intercept)

        # Calculate and append logistic loss
        y_pred_prob = model.predict_proba(X)[:, 1]
        loss = log_loss(y, y_pred_prob)
        loss_list.append(loss)

        # Plot each dataset and decision boundary
        # Plot setup
        plt.subplot(n_rows, n_cols, i)
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0', alpha=0.6)
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1', alpha=0.6)
        
        x_vals = np.linspace(x_min, x_max, 200)
        y_vals = slope * x_vals + intercept
        plt.plot(x_vals, y_vals, color='green', linestyle='--', label='Decision Boundary')
        # Set plot bounds
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        # Decision boundary
        x_vals = np.linspace(x_min, x_max, 200)
        y_vals = (-beta0 - beta1 * x_vals) / beta2
        plt.plot(x_vals, y_vals, color='green', linestyle='--', label='Decision Boundary')

        # Improve plot aesthetics
        plt.title(f"Shift Distance = {distance:.2f}", fontsize=12)
        plt.xlabel("X1", fontsize=10)
        plt.ylabel("X2", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=8)

        # Margin width calculation
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
        contour_levels = [0.7, 0.8, 0.9]
        alphas = [0.05, 0.1, 0.15]
        for level, alpha in zip(contour_levels, alphas):
            class_1_contour = plt.contourf(xx, yy, Z, levels=[level, 1.0], colors=['red'], alpha=alpha)
            class_0_contour = plt.contourf(xx, yy, Z, levels=[0.0, 1 - level], colors=['blue'], alpha=alpha)
            if level == 0.7:
                distances = cdist(class_1_contour.collections[0].get_paths()[0].vertices,
                                  class_0_contour.collections[0].get_paths()[0].vertices, metric='euclidean')
                min_distance = np.min(distances)
                margin_widths.append(min_distance)

        plt.title(f"Shift Distance = {distance}", fontsize=14)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend(loc='lower right', fontsize=8)

    plt.plot(shift_distances, beta0_list, marker='o')
    plt.tight_layout()
    plt.savefig(f"{result_dir}/dataset.png")

    print(f"shift_distances length: {len(shift_distances)}")
    print(f"beta0_list length: {len(beta0_list)}")

    # Plot results: Parameters vs. Shift Distance
    plt.figure(figsize=(18, 15))

    # Plot Beta0
    plt.subplot(3, 3, 1)
    plt.plot(shift_distances, beta0_list, marker='o')
    plt.title("Shift Distance vs Beta0")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta0")

    # Plot Beta1
    plt.subplot(3, 3, 2)
    plt.plot(shift_distances, beta1_list, marker='o')
    plt.title("Shift Distance vs Beta1 (Coefficient for x1)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta1")

    # Plot Beta2
    plt.subplot(3, 3, 3)
    plt.plot(shift_distances, beta2_list, marker='o')
    plt.title("Shift Distance vs Beta2 (Coefficient for x2)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta2")

    # Plot slope (Beta1 / Beta2)
    plt.subplot(3, 3, 4)
    plt.plot(shift_distances, slope_list, marker='o')
    plt.title("Shift Distance vs Slope (Beta1 / Beta2)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Slope")

    # Plot intercept (Beta0 / Beta2)
    plt.subplot(3, 3, 5)
    plt.plot(shift_distances, intercept_list, marker='o')
    plt.title("Shift Distance vs Intercept Ratio (Beta0 / Beta2)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Intercept")

    # Plot logistic loss
    plt.subplot(3, 3, 6)
    plt.plot(shift_distances, loss_list, marker='o')
    plt.title("Shift Distance vs Logistic Loss")
    plt.xlabel("Shift Distance")
    plt.ylabel("Logistic Loss")

    # Plot margin width
    plt.subplot(3, 3, 7)
    plt.plot(shift_distances, margin_widths, marker='o')
    plt.title("Shift Distance vs Margin Width")
    plt.xlabel("Shift Distance")
    plt.ylabel("Margin Width")

    plt.tight_layout()
    plt.savefig(f"{result_dir}/parameters_vs_shift_distance.png")

if __name__ == "__main__":
    start = 0.25
    end = 2.0
    step_num = 8
    do_experiments(start, end, step_num)
