/** Shared display constants used across sidebar, page, and config components. */

export const MODULE_LABELS: Record<string, string> = {
  numpy: "NumPy",
  sklearn: "Scikit-learn",
  pytorch: "PyTorch",
};

export const TASK_LABELS: Record<string, string> = {
  backpropagation: "Backpropagation",
  fundamentals: "Fundamentals",
  q_learning: "Q-Learning",
  attention: "Self-Attention",
  linear_regression: "Linear Regression",
  logistic_regression: "Logistic Regression",
  knn: "K-Nearest Neighbors",
  decision_tree: "Decision Tree",
  svm: "SVM",
  random_forest: "Random Forest",
  kmeans: "K-Means",
  pca: "PCA",
  xgboost: "XGBoost",
  tabular_classification: "Tabular Classification",
  image_classification: "Image Classification",
  text_classification: "Text Classification",
  time_series_forecasting: "Time Series Forecasting",
  fine_tuning: "LoRA Fine-Tuning",
  question_answering: "Question Answering",
  cnn: "CNN",
  gan: "Generative Adversarial Network",
  lstm: "LSTM Text Generation",
};

export const TASK_PLOTS: Record<string, string[]> = {
  "numpy/backpropagation": ["backpropagation_results.png"],
  "numpy/q_learning": ["q_learning_grid_path.png", "q_learning_policy_map.png"],
  "numpy/attention": ["attention_weights_matrix.png"],
  "sklearn/linear_regression": ["linear_regression_results.png"],
  "sklearn/logistic_regression": ["logistic_regression_confusion_matrix.png"],
  "sklearn/knn": ["knn_confusion_matrix.png", "knn_accuracy_vs_k.png"],
  "sklearn/decision_tree": [
    "decision_tree_visualization.png",
    "decision_tree_confusion_matrix.png",
  ],
  "sklearn/svm": ["svm_confusion_matrix.png", "svm_accuracy_vs_kernel.png"],
  "sklearn/random_forest": [
    "random_forest_confusion_matrix.png",
    "random_forest_feature_importance.png",
  ],
  "sklearn/kmeans": ["kmeans_clustering_results.png", "kmeans_elbow_plot.png"],
  "sklearn/pca": ["pca_results.png", "pca_loadings.png", "pca_reconstruction_error.png"],
  "sklearn/xgboost": ["xgboost_confusion_matrix.png", "xgboost_feature_importance.png"],
  "pytorch/cnn": ["cnn_feature_activations.png", "cnn_confusion_matrix.png"],
  "pytorch/gan": ["gan_distribution_scatter.png", "gan_loss_curves.png"],
  "pytorch/lstm": ["lstm_token_probabilities.png"],
};
