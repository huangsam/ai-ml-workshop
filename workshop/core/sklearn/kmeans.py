"""
K-Means Clustering Example using Iris Dataset

This script demonstrates K-Means clustering, an unsupervised algorithm
that partitions data into k clusters by minimizing within-cluster variance.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    # Step 1: Load and prepare the dataset
    # Iris dataset: measurements of sepal/petal lengths/widths for 3 flower species
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y_true = pd.Series(iris.target, name="species")  # True labels (for comparison only)

    print("Iris Dataset:")
    print(f"Features: {list(X.columns)}")
    print(f"Dataset shape: {X.shape}")
    print(f"Species: {iris.target_names}")
    print()

    # Step 2: Define Pipeline and Hyperparameter Grid
    # Using a Pipeline is best practice to prevent data leakage during cross-validation.
    # Clustering is sensitive to scales, so standardize features.
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),  # Preprocessing step
            ("kmeans", KMeans(random_state=42)),  # Model step
        ]
    )

    # Parameter grid for the pipeline steps
    # Note the double underscore notation: 'step_name__parameter_name'
    param_grid = {"kmeans__n_clusters": [2, 3, 4, 5, 6]}

    # Custom scorer for silhouette score
    def silhouette_scorer(estimator, X):
        labels = estimator.fit_predict(X)
        if len(np.unique(labels)) > 1:
            return silhouette_score(X, labels)
        return -1  # Invalid for single cluster

    # GridSearchCV will now cross-validate the entire pipeline
    search = GridSearchCV(pipeline, param_grid, cv=3, scoring=silhouette_scorer)
    search.fit(X)

    # Best model (it's a fitted pipeline)
    best_pipeline = search.best_estimator_
    print("Best hyperparameters found:")
    print(search.best_params_)
    print()

    # Step 4: Perform clustering
    # Access the KMeans model inside the pipeline
    final_model = best_pipeline.named_steps["kmeans"]
    labels = final_model.fit_predict(best_pipeline.named_steps["scaler"].transform(X))  # Assign clusters
    centroids = final_model.cluster_centers_  # Cluster centers

    print("Clustering Results:")
    print(f"Number of clusters: {final_model.n_clusters}")
    print(f"Cluster centers (scaled):\n{centroids}")
    print(f"Cluster assignments: {np.bincount(labels)}")  # Count per cluster
    print()

    # Step 5: Evaluate clustering quality
    X_scaled = best_pipeline.named_steps["scaler"].transform(X)
    silhouette_avg = silhouette_score(X_scaled, labels)
    print("Clustering Evaluation:")
    print(f"Silhouette Score: {silhouette_avg:.4f}")  # Higher is better (closer to 1)
    print("(Values > 0.5 indicate reasonable clustering)")
    print()

    # Step 6: Compare with true labels (for educational purposes)
    # In real unsupervised learning, we don't have true labels
    contingency_table = pd.crosstab(y_true, labels, rownames=["True Species"], colnames=["Predicted Cluster"])
    print("Contingency Table (True Species vs Clusters):")
    print(contingency_table)
    print()

    # Step 7: Visualize results
    # Plot clusters in 2D using first two features
    plt.figure(figsize=(12, 5))

    # Subplot 1: Clusters
    plt.subplot(1, 2, 1)
    X_scaled = best_pipeline.named_steps["scaler"].transform(X)  # Get scaled features
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap="viridis", alpha=0.7)
    plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="x", s=200, linewidth=3, label="Centroids")
    plt.xlabel("Sepal Length (scaled)")
    plt.ylabel("Sepal Width (scaled)")
    plt.title("K-Means Clusters")
    plt.legend()
    plt.colorbar(scatter)

    # Subplot 2: True labels for comparison
    plt.subplot(1, 2, 2)
    scatter_true = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_true, cmap="viridis", alpha=0.7)
    plt.xlabel("Sepal Length (scaled)")
    plt.ylabel("Sepal Width (scaled)")
    plt.title("True Species Labels")
    plt.colorbar(scatter_true)

    plt.tight_layout()
    plt.savefig("kmeans_clustering_results.png", dpi=300, bbox_inches="tight")
    print("Clustering visualization saved as 'kmeans_clustering_results.png'")

    # Elbow plot for different k values
    k_values = range(1, 11)
    inertias = []

    for k in k_values:
        kmeans_temp = KMeans(n_clusters=k, random_state=42)
        kmeans_temp.fit(X_scaled)
        inertias.append(kmeans_temp.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker="o", linestyle="-", color="b")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Within-cluster Sum of Squares)")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True)
    plt.savefig("kmeans_elbow_plot.png", dpi=300, bbox_inches="tight")
    print("Elbow plot saved as 'kmeans_elbow_plot.png'")


if __name__ == "__main__":
    main()
