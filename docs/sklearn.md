# Machine Learning Model Selection & Sklearn Algorithms

This document covers model selection strategies and detailed sklearn algorithm implementations. Learn which algorithms work best for different data types and problems, then dive into 8 complete algorithm examples organized by learning type.

## Model Selection Guide

Choose your algorithm based on your problem type, data characteristics, and constraints:

| Model Type      | Best For                        | Data Type         | Pros                | Cons                | Examples |
|-----------------|---------|----------|----------|---------|----------|
| **Logistic/Linear Regression** | Simple classification/regression | Tabular | Fast, interpretable | Limited complexity | sklearn |
| **Decision Trees** | Interpretable decisions, feature importance | Tabular | Human-readable rules, fast | Can overfit easily | sklearn |
| **Random Forests** | Tabular, high accuracy, feature importance | Tabular | Fast, accurate, interpretable | Less flexible | sklearn |
| **SVM** | Complex patterns, non-linear boundaries | Tabular, text | Powerful, works in high dimensions | Slow on large data | sklearn |
| **KNN** | Pattern recognition, anomaly detection | Tabular | Simple, no training | Slow at prediction, needs scaling | sklearn |
| **K-Means** | Customer segmentation, clustering | Tabular | Fast, interpretable | Needs k specified | sklearn |
| **PCA** | Dimensionality reduction, visualization | Tabular | Reduces features, faster training | Loses interpretability | sklearn |
| **Neural Networks** | Vision, audio, complex tabular | Images, audio, time series | Flexible, powerful | Needs lots of data | pytorch |
| **LLMs** | Advanced NLP, text understanding | Unstructured text | Contextual, nuanced | Resource intensive, expensive | pytorch |

### Decision Logic
1. **Tabular data**: Start with Random Forest or SVM for accuracy, Linear/Logistic Regression for speed/interpretability
2. **Image data**: Use CNNs (ResNet, VGG) or Vision Transformers
3. **Sequential data**: Use RNNs (LSTM, GRU) or Transformers
4. **Text data**: Use BERT, GPT, or other LLMs for deep understanding
5. **Unsupervised**: PCA for dimensionality reduction, K-Means for clustering

---

## Sklearn Algorithm Reference

This section summarizes the sklearn examples in the `examples/sklearn/` folder, organized by learning type (supervised vs unsupervised) and ordered from least complex to most complex. Each algorithm includes a brief description, key concepts, use cases, and reference to the example file.

## Supervised Learning Algorithms

Supervised learning uses labeled data to train models that can predict outcomes for new, unseen data.

### 1. Linear Regression (`linear_regression.py`)
**Complexity**: Low
**Type**: Regression
**Description**: Fits a straight line to predict continuous numerical values.
**Key Concepts**:
- Least squares optimization
- Coefficients represent feature importance
- R² score measures goodness of fit
**Use Cases**: House price prediction, sales forecasting, any continuous prediction
**Hyperparameters**: `fit_intercept` (whether to include y-intercept)

### 2. Logistic Regression (`logistic_regression.py`)
**Complexity**: Low-Medium
**Type**: Binary Classification
**Description**: Uses sigmoid function to predict probabilities of class membership.
**Key Concepts**:
- Log-odds transformation
- Regularization (L1/L2) prevents overfitting
- Decision boundary is linear in feature space
**Use Cases**: Spam detection, medical diagnosis, credit scoring
**Hyperparameters**: `C` (regularization strength), `penalty` (L1/L2), `solver` (optimization algorithm)

### 3. K-Nearest Neighbors (KNN) (`knn.py`)
**Complexity**: Medium
**Type**: Classification (can be adapted for regression)
**Description**: Classifies based on majority vote of k nearest training examples.
**Key Concepts**:
- Distance metrics (Euclidean, Manhattan)
- Lazy learning (no training phase)
- Feature scaling is crucial
**Use Cases**: Pattern recognition, recommendation systems, anomaly detection
**Hyperparameters**: `n_neighbors` (k value), `weights` (uniform/distance), `metric` (distance function)

### 4. Decision Tree (`decision_tree.py`)
**Complexity**: Medium
**Type**: Classification (can be adapted for regression)
**Description**: Builds tree by recursively splitting data based on feature values.
**Key Concepts**:
- Information gain/entropy reduction
- Pruning prevents overfitting
- Interpretable rules
**Use Cases**: Customer segmentation, medical diagnosis, fraud detection
**Hyperparameters**: `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion` (gini/entropy)

### 5. Support Vector Machine (SVM) (`svm.py`)
**Complexity**: Medium-High
**Type**: Classification (can be adapted for regression)
**Description**: Finds optimal hyperplane that maximizes margin between classes.
**Key Concepts**:
- Kernel trick for non-linear boundaries
- Support vectors define the decision boundary
- C parameter controls margin violations
**Use Cases**: Text classification, image recognition, bioinformatics
**Hyperparameters**: `C` (regularization), `gamma` (kernel coefficient), `kernel` (linear/rbf/poly)

### 6. Random Forest (`random_forest.py`)
**Complexity**: High
**Type**: Ensemble Classification (can be adapted for regression)
**Description**: Ensemble of decision trees with bagging and feature randomization.
**Key Concepts**:
- Bootstrap aggregating reduces variance
- Feature importance built-in
- Handles missing values and outliers well
**Use Cases**: Any classification task where accuracy is prioritized, feature selection
**Hyperparameters**: `n_estimators` (number of trees), `max_depth`, `min_samples_split`, `bootstrap`

## Unsupervised Learning Algorithms

Unsupervised learning finds patterns in data without labeled examples.

### 1. Principal Component Analysis (PCA) (`pca.py`)
**Complexity**: Medium
**Type**: Dimensionality Reduction
**Description**: Linear transformation that projects data onto principal components capturing maximum variance.
**Key Concepts**:
- Eigenvalue decomposition
- Explained variance ratio
- Component loadings show feature contributions
**Use Cases**: Data visualization, noise reduction, feature extraction, preprocessing for other algorithms
**Hyperparameters**: `n_components` (number of dimensions to keep)

### 2. K-Means Clustering (`kmeans.py`)
**Complexity**: Medium
**Type**: Clustering
**Description**: Partitions data into k clusters by minimizing within-cluster variance.
**Key Concepts**:
- Centroid-based clustering
- Elbow method for optimal k
- Silhouette score measures cluster quality
**Use Cases**: Customer segmentation, image compression, document clustering
**Hyperparameters**: `n_clusters` (k value)

## General Concepts Across All Examples

### Data Preprocessing
- **Feature Scaling**: Crucial for distance-based algorithms (KNN, SVM, K-Means) and PCA
- **Train/Test Split**: Evaluates generalization performance
- **Stratification**: Maintains class distribution in classification tasks

### Model Evaluation
- **Supervised Metrics**: Accuracy, precision/recall/F1, confusion matrix, R²/MSE
- **Unsupervised Metrics**: Silhouette score, explained variance, reconstruction error
- **Cross-Validation**: Prevents overfitting during hyperparameter tuning

### Hyperparameter Tuning
- **GridSearchCV**: Exhaustive search over parameter combinations
- **RandomizedSearchCV**: Efficient random sampling for large parameter spaces
- **Scoring Functions**: Choose appropriate metrics (accuracy, silhouette, etc.)

### Best Practices
1. **Start Simple**: Begin with basic algorithms (linear regression, logistic regression)
2. **Scale Features**: Always consider standardization/normalization
3. **Tune Hyperparameters**: Use cross-validation to optimize performance
4. **Evaluate Thoroughly**: Use multiple metrics and visualizations
5. **Compare Algorithms**: Try multiple approaches for your specific problem
6. **Interpret Results**: Understand feature importance and model decisions

### When to Choose Which Algorithm

**For Beginners**:
- Regression: Linear Regression
- Classification: Logistic Regression or KNN

**For Interpretability**:
- Decision Trees (rules are human-readable)

**For High Accuracy**:
- Random Forest or SVM (often best performers)

**For Speed**:
- Linear/Logistic Regression or KNN

**For Unsupervised Tasks**:
- Dimensionality reduction: PCA
- Clustering: K-Means

**For Complex Data**:
- SVM with kernels or ensemble methods

Remember: No single algorithm is universally best. The choice depends on your data, problem type, interpretability needs, and computational constraints. Always experiment and validate!

---

## Project 4: Classical ML Algorithms

This project implemented 8 complete examples covering supervised and unsupervised learning:

### Supervised Learning
- **Linear Regression**: California Housing dataset - predicting house prices
- **Logistic Regression**: Binary classification with regularization
- **Decision Trees**: Interpretable tree-based classification with pruning
- **Random Forest**: Ensemble method combining multiple trees for improved accuracy
- **KNN**: Distance-based classification with k-nearest neighbors
- **SVM**: Support vector machines with kernel methods for non-linear classification

### Unsupervised Learning
- **K-Means Clustering**: Customer segmentation and pattern discovery
- **PCA**: Dimensionality reduction for visualization and feature extraction

### Key Learnings
- Hyperparameter tuning with GridSearchCV and RandomizedSearchCV
- Cross-validation for robust evaluation
- Feature scaling and preprocessing importance
- Evaluation metrics: accuracy, precision, recall, F1-score, confusion matrix
- Model comparison and algorithm selection strategies
- Complete examples with visualization and educational comments

