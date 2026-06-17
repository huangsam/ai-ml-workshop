export interface TheoryContent {
  title: string;
  overview: string;
  concepts: { term: string; definition: string }[];
  math?: string;
  whatToObserve: string[];
}

export const THEORY_DATA: Record<string, TheoryContent> = {
  "numpy/fundamentals": {
    title: "NumPy Fundamentals",
    overview:
      "NumPy is the foundational package for scientific computing in Python. This lesson introduces vectorization, matrix operations, and basic linear algebra that form the mathematical backbone of deep learning.",
    concepts: [
      {
        term: "Vectorization",
        definition:
          "Writing mathematical operations that apply to entire arrays rather than looping over individual elements, leveraging highly optimized C-code under the hood.",
      },
      {
        term: "Broadcasting",
        definition:
          "The mechanism allowing arithmetic operations between arrays of different shapes during computations.",
      },
      {
        term: "Dot Product",
        definition:
          "The sum of the products of the corresponding elements of two sequences of numbers, representing projection and similarity.",
      },
    ],
    math: "Matrix Multiplication (C = A @ B):\nC[i, j] = sum(k = 0 to N-1, A[i, k] * B[k, j])",
    whatToObserve: [
      "Observe the difference in execution speed between nested loops and vectorized NumPy operations.",
      "Notice how matrix dimensions must align (inner dimensions must match) for dot products.",
    ],
  },
  "numpy/backpropagation": {
    title: "Neural Network Backpropagation",
    overview:
      "Backpropagation is the core algorithm used to train neural networks. It calculates the gradient of the loss function with respect to the weights using the chain rule, propagating errors backward from the output layer to the input layer.",
    concepts: [
      {
        term: "Forward Pass",
        definition:
          "Computing the activations of each layer sequentially to predict the output for given inputs.",
      },
      {
        term: "Loss Function",
        definition:
          "A metric quantifying the discrepancy between the network's predictions and the true labels (e.g., Mean Squared Error).",
      },
      {
        term: "Gradient Descent",
        definition:
          "An optimization algorithm that updates weights in the opposite direction of the gradient to minimize the loss function.",
      },
      {
        term: "Chain Rule",
        definition:
          "A calculus rule used to compute the derivative of composite functions, enabling step-by-step backpropagation.",
      },
    ],
    math: "Weight Update Rule:\nW = W - η * (dL / dW)\nwhere:\nη = Learning Rate\ndL/dW = Gradient of the Loss with respect to the weight W",
    whatToObserve: [
      "Watch how the training loss decreases steadily over epochs.",
      "Observe the final decision boundary plot: as training progresses, the model separates the binary classes more accurately.",
    ],
  },
  "numpy/q_learning": {
    title: "Q-Learning Maze Navigation",
    overview:
      "Q-Learning is a model-free, value-based Reinforcement Learning algorithm. The agent learns the value of performing actions in specific states to maximize cumulative future rewards, mapping out an optimal path through a grid or environment.",
    concepts: [
      {
        term: "State (S)",
        definition:
          "The current position of the agent in the environment (e.g., coordinates in a grid).",
      },
      {
        term: "Action (A)",
        definition: "The choices available to the agent (e.g., Up, Down, Left, Right).",
      },
      {
        term: "Reward (R)",
        definition:
          "Feedback from the environment (+10 for reaching the goal, -1 for each step to encourage speed, -100 for hitting obstacles).",
      },
      {
        term: "Q-Table",
        definition:
          "A lookup table storing Q-values, representing the expected future reward for each state-action pair.",
      },
    ],
    math: "Q-Value Update (Bellman Equation):\nQ(s, a) = Q(s, a) + α * [ R + γ * max_a' Q(s', a') - Q(s, a) ]\nwhere:\nα = Learning Rate\nγ = Discount Factor (importance of future rewards)\ns' = Next State",
    whatToObserve: [
      "Observe how the policy map develops arrows pointing toward the goal as training completes.",
      "Notice the path length decreasing as the agent shifts from exploration (random steps) to exploitation (following the Q-table).",
    ],
  },
  "numpy/attention": {
    title: "Self-Attention Mechanism",
    overview:
      "Self-Attention is the foundational building block of the Transformer architecture (e.g., GPT, BERT). It allows tokens in a sequence to dynamically compute weightings and context vectors based on their relations to all other tokens.",
    concepts: [
      { term: "Query (Q)", definition: "Represents what a token is looking for in the sequence." },
      {
        term: "Key (K)",
        definition: "Represents what information a token holds to match against queries.",
      },
      {
        term: "Value (V)",
        definition: "Represents the actual content vector that gets aggregated into the output.",
      },
      {
        term: "Softmax",
        definition:
          "An activation function that normalizes attention scores into a probability distribution summing to 1.",
      },
    ],
    math: "Scaled Dot-Product Attention:\nAttention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V\nwhere:\nd_k = Dimension of the key vectors",
    whatToObserve: [
      "Examine the attention weights matrix heatmap: diagonal entries represent self-focus, while off-diagonals show relations between words.",
      "Notice how changing query dimensions affects attention score magnitudes.",
    ],
  },
  "sklearn/linear_regression": {
    title: "Linear Regression",
    overview:
      "Linear Regression is a fundamental statistical algorithm that models the linear relationship between a dependent target variable and one or more independent features by fitting a straight line.",
    concepts: [
      {
        term: "Coefficient / Slope",
        definition:
          "The rate of change in the target variable for each unit change in the feature.",
      },
      {
        term: "Intercept",
        definition: "The value where the fitted line crosses the Y-axis (value when X is 0).",
      },
      {
        term: "Residuals",
        definition:
          "The vertical distances between the actual data points and the fitted regression line.",
      },
    ],
    math: "Prediction:\ny_pred = w * x + b\nObjective (Mean Squared Error):\nMSE = (1 / N) * sum( (y_actual - y_pred)^2 )",
    whatToObserve: [
      "Observe the final scatter plot showing how closely the fitted red regression line runs through the noisy data points.",
      "Look at the MSE metric: lower values signify a better line fit.",
    ],
  },
  "sklearn/logistic_regression": {
    title: "Logistic Regression",
    overview:
      "Logistic Regression is used for binary classification tasks. It models the probability that a given input belongs to a specific class using the sigmoid activation function.",
    concepts: [
      {
        term: "Sigmoid Function",
        definition:
          "An S-shaped function that maps any real-valued number into a probability between 0 and 1.",
      },
      {
        term: "Log-Odds / Logit",
        definition:
          "The logarithm of the ratio of the probability of an event occurring to the probability of it not occurring.",
      },
      {
        term: "Decision Boundary",
        definition:
          "The threshold line (typically at probability = 0.5) separating predictions into Class 0 or Class 1.",
      },
    ],
    math: "Sigmoid Activation:\nσ(z) = 1 / (1 + e^-z)\nwhere:\nz = w * x + b",
    whatToObserve: [
      "Examine the confusion matrix: high numbers on the top-left to bottom-right diagonal indicate correct classifications.",
      "Observe how the classification boundary cleanly divides the binary classes.",
    ],
  },
  "sklearn/knn": {
    title: "K-Nearest Neighbors (KNN)",
    overview:
      "KNN is a non-parametric, instance-based classification algorithm. It classifies a new data point based on the majority class of its 'K' nearest neighbors in the feature space.",
    concepts: [
      {
        term: "K Parameter",
        definition:
          "The number of nearest data points to inspect before making a prediction. Small K leads to overfitting; large K can over-smooth boundaries.",
      },
      {
        term: "Euclidean Distance",
        definition:
          "The straight-line distance between two points in Euclidean space, used to measure neighbor proximity.",
      },
      {
        term: "Curse of Dimensionality",
        definition:
          "As the number of features grows, distance metrics become less effective because data points appear equidistant.",
      },
    ],
    math: "Euclidean Distance:\nd(p, q) = sqrt( sum( (p_i - q_i)^2 ) )",
    whatToObserve: [
      "Compare accuracy across different values of K in the accuracy vs. K plot.",
      "Notice how a very small K (e.g., K=1) creates complex, jagged decision boundaries, while larger K values create smoother boundaries.",
    ],
  },
  "sklearn/decision_tree": {
    title: "Decision Tree Classifier",
    overview:
      "Decision Trees split the dataset recursively into subsets based on feature thresholds. They create a flowchart-like structure of decision rules to classify instances.",
    concepts: [
      {
        term: "Gini Impurity",
        definition:
          "A measure of how often a randomly chosen element from the set would be incorrectly labeled if it were randomly labeled according to the distribution of labels in the subset.",
      },
      {
        term: "Information Gain",
        definition:
          "The reduction in entropy or impurity achieved by splitting the data on a specific feature.",
      },
      {
        term: "Pruning",
        definition:
          "Cutting back branches of the tree to prevent overfitting and improve generalization.",
      },
    ],
    math: "Gini Impurity:\nGini = 1 - sum( p_i^2 )\nwhere:\np_i = Probability of an item belonging to class i in the split node",
    whatToObserve: [
      "Look at the Decision Tree visual diagram: notice how nodes split on features that maximize purity.",
      "Compare training vs. test accuracy: unconstrained trees tend to overfit (100% training accuracy but lower test accuracy).",
    ],
  },
  "sklearn/svm": {
    title: "Support Vector Machines (SVM)",
    overview:
      "SVM is a powerful classification algorithm that finds the optimal hyperplane maximizing the margin (distance) between different classes.",
    concepts: [
      {
        term: "Support Vectors",
        definition:
          "The data points closest to the decision boundary. These points define the orientation and position of the margin.",
      },
      {
        term: "Margin",
        definition:
          "The width of the gap between the decision boundary and the closest support vectors.",
      },
      {
        term: "Kernel Trick",
        definition:
          "Mapping input features into higher-dimensional spaces to make non-linearly separable data linearly separable.",
      },
    ],
    math: "Linear Decision Hyperplane:\nw * x + b = 0\nMargin Width:\nWidth = 2 / ||w||",
    whatToObserve: [
      "Observe the support vectors indicated on the SVM decision boundary plot.",
      "Notice how switching from a linear kernel to an RBF (radial basis function) kernel allows the boundary to wrap around complex, non-linear shapes.",
    ],
  },
  "sklearn/random_forest": {
    title: "Random Forest Classifier",
    overview:
      "Random Forest is an ensemble learning method that builds multiple decision trees during training and merges their predictions (voting) to get a more accurate and stable prediction.",
    concepts: [
      {
        term: "Bagging (Bootstrap Aggregating)",
        definition:
          "Training individual trees on random subsets of the data sampled with replacement.",
      },
      {
        term: "Feature Randomness",
        definition:
          "Forcing trees to only split on random subsets of features rather than all available features, reducing tree correlation.",
      },
      {
        term: "Out-Of-Bag (OOB) Error",
        definition:
          "A method of measuring prediction error of bootstrap samples by testing them on trees that did not contain them in training.",
      },
    ],
    math: "Ensemble Prediction (Majority Vote):\ny_pred = mode( Tree_1(x), Tree_2(x), ..., Tree_B(x) )",
    whatToObserve: [
      "Look at the Feature Importance plot: see which features are most heavily relied upon by the ensemble.",
      "Notice that the Random Forest achieves better generalization and test accuracy than a single Decision Tree.",
    ],
  },
  "sklearn/kmeans": {
    title: "K-Means Clustering",
    overview:
      "K-Means is an unsupervised learning algorithm that partitions data points into 'K' distinct, non-overlapping clusters based on spatial proximity.",
    concepts: [
      {
        term: "Centroid",
        definition:
          "The mathematical center or mean coordinate of all data points assigned to a cluster.",
      },
      {
        term: "Inertia (Within-Cluster Sum of Squares)",
        definition:
          "The sum of squared distances of samples to their closest cluster center, representing cluster tightness.",
      },
      {
        term: "Elbow Method",
        definition:
          "Plotting inertia against the number of clusters to find the optimal 'K' where the rate of inertia decrease levels off.",
      },
    ],
    math: "Centroid Update:\nμ_k = (1 / |S_k|) * sum( x_i )\nwhere:\nS_k = Set of points assigned to centroid k",
    whatToObserve: [
      "Watch the centroids adjust iteratively to align with the density centers of the data points.",
      "Check the Elbow Plot to identify the 'kink' representing the natural number of clusters in the dataset.",
    ],
  },
  "sklearn/pca": {
    title: "Principal Component Analysis (PCA)",
    overview:
      "PCA is an unsupervised dimensionality reduction technique. It projects high-dimensional data onto orthogonal directions (principal components) that maximize variance, compressing data while retaining information.",
    concepts: [
      {
        term: "Principal Components",
        definition:
          "New, uncorrelated orthogonal features created as linear combinations of original features.",
      },
      {
        term: "Eigenvalues & Eigenvectors",
        definition:
          "Mathematical constructs derived from the covariance matrix representing the direction (eigenvectors) and magnitude (eigenvalues) of data variance.",
      },
      {
        term: "Explained Variance Ratio",
        definition: "The percentage of total variance captured by each principal component.",
      },
    ],
    math: "Covariance Matrix (Σ):\nΣ = (1 / N) * X.T @ X\nProjection:\nX_projected = X @ W\nwhere:\nW = top eigenvectors of Σ",
    whatToObserve: [
      "Observe the 2D projection scatter plot: high-dimensional classes should cluster separately even when compressed.",
      "Check the cumulative variance curve: observe how many components are needed to explain 90%+ of the dataset's variance.",
    ],
  },
  "sklearn/xgboost": {
    title: "XGBoost Classifier",
    overview:
      "XGBoost (Extreme Gradient Boosting) is an optimized, highly efficient implementation of gradient boosted decision trees. It builds trees sequentially, with each tree correcting the errors of the preceding trees.",
    concepts: [
      {
        term: "Boosting",
        definition:
          "An ensemble technique where models are trained sequentially, focusing on instances that were misclassified by earlier models.",
      },
      {
        term: "Gradient Descent in Loss",
        definition:
          "Building trees that predict the negative gradients of the loss function, moving predictions closer to the targets.",
      },
      {
        term: "Regularization",
        definition:
          "L1 (Lasso) and L2 (Ridge) penalties on leaf weights to control tree complexity and prevent overfitting.",
      },
    ],
    math: "Sequential Model Update:\nF_t(x) = F_{t-1}(x) + η * f_t(x)\nwhere:\nf_t(x) = tree trained to fit residuals/gradients of step t-1",
    whatToObserve: [
      "Examine the feature importances to see which factors drive XGBoost classifications.",
      "Note the high test accuracy compared to standard decision trees: boosting typically yields state-of-the-art results on tabular data.",
    ],
  },
  "pytorch/tabular_classification": {
    title: "Tabular Classification with PyTorch",
    overview:
      "This lesson trains a PyTorch Multi-Layer Perceptron (MLP) neural network on structured tabular data, implementing custom dataset loading, dense linear layers, and binary/multi-class cross-entropy optimization.",
    concepts: [
      {
        term: "Linear Layer",
        definition:
          "A fully connected layer applying matrix multiplication and bias offset (y = xW.T + b).",
      },
      {
        term: "Cross-Entropy Loss",
        definition:
          "The standard loss function for classification, measuring similarity between predicted probability distributions and true one-hot distributions.",
      },
      {
        term: "Adam Optimizer",
        definition:
          "An extension to stochastic gradient descent that computes adaptive learning rates for each parameter based on first and second moments of gradients.",
      },
    ],
    math: "Cross-Entropy Loss:\nL = - (1 / N) * sum( y_i * log(p_i) + (1 - y_i) * log(1 - p_i) )",
    whatToObserve: [
      "Observe the training vs. validation loss curve: validation loss should decrease in step with training loss.",
      "Look for divergence: if validation loss begins rising while training loss continues falling, the model is overfitting.",
    ],
  },
  "pytorch/image_classification": {
    title: "Image Classification with PyTorch",
    overview:
      "This lesson trains a neural network to recognize structures within synthetic image grids, applying multi-dimensional matrix operations and linear projections directly to raw pixel values.",
    concepts: [
      {
        term: "Pixel Flattening",
        definition:
          "Converting a 2D image matrix of shape (H, W) or 3D tensor (C, H, W) into a 1D vector of size C*H*W to feed into dense layers.",
      },
      {
        term: "Batch Normalization",
        definition:
          "Normalizing the activations of a layer across the batch to stabilize training and accelerate convergence.",
      },
      {
        term: "ReLU Activation",
        definition:
          "Rectified Linear Unit function: f(x) = max(0, x), which introduces non-linearities essential for complex pattern recognition.",
      },
    ],
    math: "ReLU Activation:\nReLU(x) = max(0, x)",
    whatToObserve: [
      "Observe the final accuracy percentage: fully connected networks can perform basic image recognition but struggle with translation invariance.",
      "Notice how increasing the number of hidden units improves initial capacity.",
    ],
  },
  "pytorch/text_classification": {
    title: "Text Classification & Embeddings",
    overview:
      "This lesson demonstrates text representation and classification. Text sequences are converted into dense vector representations (embeddings) and classified based on semantic meaning.",
    concepts: [
      {
        term: "Tokenization",
        definition: "Splitting raw text strings into individual tokens (words or sub-words).",
      },
      {
        term: "Embedding Layer",
        definition:
          "A lookup table that maps high-dimensional discrete tokens into dense, continuous vector spaces representing word similarity.",
      },
      {
        term: "Global Average Pooling",
        definition:
          "Averaging embedding vectors across the sequence dimension to compress a variable-length sequence into a fixed-length document vector.",
      },
    ],
    math: "Embedding Lookup:\nE = EmbeddingTable[token_id]\nAverage Pooling:\nv_doc = (1 / T) * sum( E_t )",
    whatToObserve: [
      "Look at how classification accuracy increases as vocabulary indexing and embeddings stabilize.",
      "Observe the impact of maximum sequence length on classification speed and training throughput.",
    ],
  },
  "pytorch/time_series_forecasting": {
    title: "Time Series Forecasting",
    overview:
      "Time series forecasting trains models to predict future values based on past observations. This is critical for demand planning, financial forecasting, and signal processing.",
    concepts: [
      {
        term: "Autoregression",
        definition: "Using past steps of a variable to predict future steps of the same variable.",
      },
      {
        term: "Lookback Window",
        definition:
          "The length of past history passed as features to the network (e.g., using the past 30 days to predict tomorrow).",
      },
      {
        term: "Mean Absolute Error (MAE)",
        definition:
          "The average of the absolute differences between predictions and actual values, representing typical prediction error.",
      },
    ],
    math: "Lookback Predictor:\ny_t = f( y_{t-1}, y_{t-2}, ..., y_{t-w} )\nMAE = (1 / N) * sum( |y_actual - y_pred| )",
    whatToObserve: [
      "Compare the actual time series curve against the model's predicted forecast: look for matching phase and trend amplitudes.",
      "Notice how prediction error tends to accumulate as the forecast horizon extends.",
    ],
  },
  "pytorch/fine_tuning": {
    title: "LoRA Parameter-Efficient Fine-Tuning",
    overview:
      "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning (PEFT) method. Instead of updating all parameters of a pre-trained model, it updates low-rank decompositions of parameter update matrices, drastically reducing GPU memory and storage footprint.",
    concepts: [
      {
        term: "Pre-trained Weights",
        definition:
          "Large model parameters that are frozen during LoRA training to preserve foundational knowledge.",
      },
      {
        term: "Low-Rank Matrices (A and B)",
        definition:
          "Small adapters of rank 'r' inserted alongside layers. A projects inputs to a lower dimension, and B projects them back.",
      },
      {
        term: "Adapter Rank (r)",
        definition:
          "The bottleneck dimension of the adapters. A smaller rank updates fewer parameters, while a larger rank increases capacity.",
      },
    ],
    math: "LoRA Parameter Adaptation:\nW_updated = W_frozen + ΔW\nΔW = (α / r) * (B @ A)\nwhere:\nB is of shape (d, r) and A is of shape (r, k) with r << min(d, k)",
    whatToObserve: [
      "Compare trainable parameters: LoRA typically reduces trainable parameter count by 99% compared to full fine-tuning.",
      "Observe that the adapter training loss drops rapidly without disrupting the pre-trained weights.",
    ],
  },
  "pytorch/question_answering": {
    title: "Question Answering Model",
    overview:
      "Question Answering models extract answers from a reference text passage based on a question. The model predicts the probability distributions of the start and end positions of the answer span.",
    concepts: [
      {
        term: "Span Extraction",
        definition:
          "Identifying the start and end indices of the answer substring directly inside the context passage.",
      },
      {
        term: "Context vs. Question",
        definition:
          "The input consists of two texts concatenated together, separated by special structural tokens.",
      },
      {
        term: "Logits for Positions",
        definition:
          "The model outputs two values per token: one for its probability of being the start token, and one for the end token.",
      },
    ],
    math: "Span Probability:\nP(start = i) = softmax( StartLogits_i )\nP(end = j) = softmax( EndLogits_j )",
    whatToObserve: [
      "Observe the loss curve decreasing as the model learns to identify answer spans in text datasets.",
      "Verify that changing the context passage changes the index range predicted by the model.",
    ],
  },
  "pytorch/cnn": {
    title: "Convolutional Neural Networks (CNN)",
    overview:
      "CNNs are highly effective for image processing. They utilize spatial convolutions that apply slide filters across images to capture localized patterns like edges, textures, and shape features.",
    concepts: [
      {
        term: "Convolution Filter",
        definition:
          "A small matrix of weights (e.g., 3x3) slid across the input to compute dot products, creating activation maps.",
      },
      {
        term: "Pooling (Max Pooling)",
        definition:
          "Downsampling operation that extracts the maximum value within a window, reducing spatial dimension and providing translation invariance.",
      },
      {
        term: "Feature Maps",
        definition:
          "Intermediate activations representing spatial patterns detected by filters at different depths.",
      },
    ],
    math: "2D Convolution:\nS(i, j) = sum_m sum_n ( I(i - m, j - n) * K(m, n) )\nwhere:\nI = Input Image, K = Filter Kernel",
    whatToObserve: [
      "Look at the CNN feature activation maps: early layers show simple shapes and edges, while later layers detect complex combinations.",
      "Observe the confusion matrix to see which shapes (e.g., squares vs. triangles) are most commonly confused by the network.",
    ],
  },
  "pytorch/gan": {
    title: "Generative Adversarial Networks (GAN)",
    overview:
      "GANs consist of two neural networks competing in a zero-sum game: a Generator that creates fake data points, and a Discriminator that attempts to distinguish real data from fake data.",
    concepts: [
      {
        term: "Generator",
        definition:
          "A network taking random noise as input and learning to generate realistic data structures to fool the Discriminator.",
      },
      {
        term: "Discriminator",
        definition:
          "A classifier network trained to identify whether inputs are authentic (real) or synthetic (fake).",
      },
      {
        term: "Adversarial Equilibrium",
        definition:
          "The point where the Generator produces perfect representations and the Discriminator can only guess with 50% accuracy.",
      },
    ],
    math: "Minimax Objective:\nmin_G max_D V(D, G) = E[ log D(x) ] + E[ log(1 - D(G(z))) ]",
    whatToObserve: [
      "Examine the 2D distribution scatter plot: watch how the generated points shift and stretch to match the circular ring shape of the real data.",
      "Watch the Generator and Discriminator loss curves: they fluctuate as the two models try to outperform each other.",
    ],
  },
  "pytorch/lstm": {
    title: "LSTM Text Generation",
    overview:
      "Long Short-Term Memory (LSTM) is a recurrent neural network (RNN) architecture designed to model temporal dependencies. It uses a gating mechanism to prevent vanishing gradients, allowing the network to retain text context over long sequence horizons.",
    concepts: [
      {
        term: "Cell State (C_t)",
        definition:
          "The long-term memory corridor of the LSTM, modified only by linear interactions to retain information easily.",
      },
      {
        term: "Gates (Forget, Input, Output)",
        definition:
          "Sigmoidal regulators that decide what to erase, what to write, and what to output from the cell state.",
      },
      {
        term: "Temperature Sampling",
        definition:
          "Scaling logits during generation to control randomness. Lower temperature produces predictable text; higher temperature produces creative/chaotic text.",
      },
    ],
    math: "Cell State Update:\nC_t = f_t * C_{t-1} + i_t * tanh( W * [h_{t-1}, x_t] + b )\nwhere:\nf_t = Forget Gate activation\ni_t = Input Gate activation",
    whatToObserve: [
      "Read the sampled text outputs generated at completion: note how spelling and syntax coherence improve as loss drops.",
      "Look at the token probability distribution bar chart: see which characters are predicted as most likely next choices given a prompt.",
    ],
  },
  "pytorch/quantization": {
    title: "PyTorch Model Quantization",
    overview:
      "Model Quantization compresses neural network parameters from floating point representation (typically FP32) to lower-precision integers (typically INT8), accelerating inference speed and reducing size with minimal loss in accuracy.",
    concepts: [
      {
        term: "FP32 vs. INT8",
        definition:
          "FP32 uses 32 bits of storage per weight, allowing wide dynamic range. INT8 uses only 8 bits, reducing size by 75%.",
      },
      {
        term: "Dynamic Quantization",
        definition:
          "Quantizing weights offline (before execution) while dynamically scaling activations at runtime.",
      },
      {
        term: "Quantization Scale & Zero-Point",
        definition:
          "The scale factor and offset used to map real-valued floats to integer indexes.",
      },
    ],
    math: "Quantization Mapping:\nq = round( r / S ) + Z\nwhere:\nr = Real float value\nS = Scale factor\nZ = Zero-point integer",
    whatToObserve: [
      "Check the performance bar chart comparing FP32 vs. INT8 models.",
      "Observe that the INT8 quantized model size is ~4x smaller and latency is significantly lower, while accuracy remains almost identical.",
    ],
  },
};
