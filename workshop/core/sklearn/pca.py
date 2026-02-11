"""
Principal Component Analysis (PCA) Example using Iris Dataset

This script demonstrates PCA, a dimensionality reduction technique
that transforms data into principal components capturing maximum variance.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    # Step 1: Load and prepare the dataset
    # Iris dataset: 4 features, we'll reduce to 2 or 3 principal components
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="species")

    print("Iris Dataset:")
    print(f"Features: {list(X.columns)}")
    print(f"Dataset shape: {X.shape}")
    print(f"Species: {iris.target_names}")
    print()

    # Step 2: Define Pipeline
    # Using a Pipeline is best practice to prevent data leakage.
    # PCA is affected by scales, so we standardize features first.
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),  # Preprocessing step
            ("pca", PCA(n_components=2)),  # Dimensionality reduction step
        ]
    )

    # Step 3: Fit the pipeline and apply PCA
    X_pca = pipeline.fit_transform(X)

    print("PCA Results:")
    print(f"Original dimensions: {X.shape[1]}")
    print(f"Reduced dimensions: {X_pca.shape[1]}")
    pca = pipeline.named_steps["pca"]
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
    print()

    # Step 4: Analyze explained variance
    # Fit PCA with all components to see variance explained
    scaler = pipeline.named_steps["scaler"]
    X_scaled = scaler.transform(X)
    pca_full = PCA()
    pca_full.fit(X_scaled)

    explained_variance = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    print("Explained Variance Analysis:")
    for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance), 1):
        print(f"PC{i}: {var:.4f} ({cum_var:.4f} cumulative)")
    print()

    # Step 5: Visualize results
    # Plot the reduced 2D data colored by species
    plt.figure(figsize=(12, 5))

    # Subplot 1: PCA scatter plot
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.7)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA: First Two Components")
    plt.colorbar(scatter, ticks=[0, 1, 2], label="Species")
    plt.clim(-0.5, 2.5)  # Set colorbar limits

    # Subplot 2: Explained variance plot
    plt.subplot(1, 2, 2)
    components = range(1, len(explained_variance) + 1)
    plt.bar(components, explained_variance, alpha=0.7, label="Individual")
    plt.plot(components, cumulative_variance, marker="o", color="red", label="Cumulative")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Explained Variance by Components")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pca_results.png", dpi=300, bbox_inches="tight")
    print("PCA visualization saved as 'pca_results.png'")

    # Step 6: Component loadings (feature contributions)
    # Show which original features contribute to each component
    loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=X.columns)

    print("Principal Component Loadings:")
    print(loadings)
    print()

    # Visualize loadings as heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(loadings, annot=True, cmap="coolwarm", center=0, xticklabels=["PC1", "PC2"], yticklabels=X.columns)
    plt.title("PCA Component Loadings")
    plt.savefig("pca_loadings.png", dpi=300, bbox_inches="tight")
    print("Loadings heatmap saved as 'pca_loadings.png'")

    # Step 7: Reconstruction error analysis
    # Show how much information is lost with fewer components
    reconstruction_errors = []
    for n_comp in range(1, X.shape[1] + 1):
        pca_temp = PCA(n_components=n_comp)
        X_reduced = pca_temp.fit_transform(X_scaled)
        X_reconstructed = pca_temp.inverse_transform(X_reduced)
        error = np.mean((X_scaled - X_reconstructed) ** 2)
        reconstruction_errors.append(error)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, X.shape[1] + 1), reconstruction_errors, marker="o", linestyle="-", color="green")
    plt.xlabel("Number of Components")
    plt.ylabel("Mean Squared Reconstruction Error")
    plt.title("Reconstruction Error vs Number of Components")
    plt.grid(True)
    plt.savefig("pca_reconstruction_error.png", dpi=300, bbox_inches="tight")
    print("Reconstruction error plot saved as 'pca_reconstruction_error.png'")


if __name__ == "__main__":
    main()
