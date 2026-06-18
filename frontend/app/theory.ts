export interface TheoryContent {
  title: string;
  overview: string;
  analogy?: string;
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
    analogy:
      "Imagine a factory assembly line. Each worker (row in A) reads from a parts shelf (columns in B) and combines what they find into a finished product (cell in C). Vectorization means the entire factory runs at once — no waiting for one worker to finish before the next starts.",
    math: "Matrix Multiplication ($C = A \\times B$):\n$$C_{i, j} = \\sum_{k=0}^{N-1} A_{i, k} \\cdot B_{k, j}$$",
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
    analogy:
      "Think of a chef adjusting a recipe after tasting the dish. The 'loss' is how bad it tastes. Backprop is the chef working backwards — too salty overall? Figure out which step added the most salt, and turn it down a little. Repeat after every meal until it's perfect.",
    math: "Weight Update Rule:\n$$W \\leftarrow W - \\eta \\cdot \\frac{\\partial L}{\\partial W}$$\nwhere:\n* $\\eta$ = Learning Rate\n* $\\frac{\\partial L}{\\partial W}$ = Gradient of the Loss with respect to weight $W$",
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
    analogy:
      "Imagine a rat in a maze that earns a gold star for reaching the cheese. At first it wanders randomly. Every time it finds a good path, it mentally upgrades its 'rating' for that corridor. Over hundreds of runs, it builds a complete mental map of which direction is worth taking from every junction — no map needed upfront.",
    math: "Q-Value Update (Bellman Equation):\n$$Q(s, a) \\leftarrow Q(s, a) + \\alpha \\left[ R + \\gamma \\max_{a'} Q(s', a') - Q(s, a) \\right]$$\nwhere:\n* $\\alpha$ = Learning Rate\n* $\\gamma$ = Discount Factor (importance of future rewards)\n* $s'$ = Next State",
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
    analogy:
      "Imagine you're in a meeting and someone says 'the bank was steep.' Your brain immediately scans back through the conversation — were we talking about rivers or finance? Every word you've heard becomes a 'witness' you silently poll, weighted by how relevant they are. Self-attention is that mental scan happening simultaneously for every word in the sentence.",
    math: "Scaled Dot-Product Attention:\n$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left( \\frac{Q K^T}{\\sqrt{d_k}} \\right) V$$\nwhere:\n* $d_k$ = Dimension of the key/query vectors",
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
    analogy:
      "Think of stretching a rubber band through a cloud of dots on graph paper. You want to position it so it runs as close as possible to all the dots at once. The rubber band is your line (y = wx + b), and the MSE measures how far the dots are from the band on average.",
    math: "Prediction:\n$$\\hat{y} = w x + b$$\nObjective (Mean Squared Error):\n$$\\text{MSE} = \\frac{1}{N} \\sum_{i=1}^{N} (y_i - \\hat{y}_i)^2$$",
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
    analogy:
      "Logistic regression is like a spam filter's confidence dial. It takes any raw score — no matter how extreme — and squishes it into a probability between 0 and 1. Above 0.5 and the email goes to spam. The sigmoid is the squishing mechanism.",
    math: "Sigmoid Activation:\n$$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$\nwhere:\n$$z = w x + b$$",
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
    analogy:
      "You've moved to a new city and want restaurant recommendations. Instead of reading reviews, you find the 5 people who have the most similar taste history to yours and ask what they liked. KNN does the same thing — but 'nearby' means similar feature values, not physical proximity.",
    math: "Euclidean Distance:\n$$d(p, q) = \\sqrt{\\sum_{i=1}^{d} (p_i - q_i)^2}$$",
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
    analogy:
      "A decision tree plays a game of 20 Questions. At each step it asks the single question that most cleanly divides what's left — 'Is it an animal?' before 'Does it have spots?' Gini impurity measures how 'mixed' the remaining pile is. A perfect split leaves one pile of all cats and another of all dogs — zero impurity.",
    math: "Gini Impurity:\n$$\\text{Gini} = 1 - \\sum_{i=1}^{C} p_i^2$$\nwhere:\n* $p_i$ = Probability of an item belonging to class $i$ in the split node",
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
    analogy:
      "Imagine trying to separate red and blue marbles on a table by placing a ruler between them. SVM finds the ruler position that leaves the biggest possible gap on each side. The marbles touching the ruler's exclusion zone are the 'support vectors' — they're the only ones that matter for deciding where the ruler goes.",
    math: "Linear Decision Hyperplane:\n$$w \\cdot x + b = 0$$\nMargin Width:\n$$\\text{Width} = \\frac{2}{\\|w\\|}$$",
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
    analogy:
      "Instead of trusting one doctor's diagnosis, you get a second opinion — then a third, fourth, and hundredth. Each doctor (decision tree) was trained on a slightly different set of patient records. The final diagnosis is a majority vote. No single doctor is perfect, but together they're much harder to fool.",
    math: "Ensemble Prediction (Majority Vote):\n$$\\hat{y} = \\text{mode}\\left( T_1(x), T_2(x), \\dots, T_B(x) \\right)$$\nwhere:\n* $T_b(x)$ = Prediction of the $b$-th decision tree",
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
    analogy:
      "Imagine sorting a bag of mixed Lego bricks by color, but you can't look at them directly — only measure distances. You guess three pile locations, assign each brick to its nearest pile, then move each pile center to the average position of its bricks. Repeat until no brick switches piles. That's K-Means.",
    math: "Centroid Update:\n$$\\mu_k = \\frac{1}{|S_k|} \\sum_{x_i \\in S_k} x_i$$\nwhere:\n* $S_k$ = Set of data points assigned to centroid $k$",
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
    analogy:
      "You have a 3D sculpture and need to photograph it in 2D. PCA finds the single angle that captures the most detail — the view where the sculpture looks the least 'squished'. Each additional principal component is the next best angle, perpendicular to the first.",
    math: "Covariance Matrix ($\\Sigma$):\n$$\\Sigma = \\frac{1}{N} X^T X$$\nProjection:\n$$X_{\\text{projected}} = X W$$\nwhere:\n* $W$ = Matrix containing top eigenvectors of $\\Sigma$ as columns",
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
    analogy:
      "Think of a team of junior analysts where each one is hired specifically to fix the mistakes of the person before them. Analyst 1 makes a rough prediction. Analyst 2 only looks at where Analyst 1 was wrong. Analyst 3 fixes what Analyst 2 missed. The final answer is everyone's corrections stacked on top of each other.",
    math: "Sequential Model Update:\n$$F_t(x) = F_{t-1}(x) + \\eta f_t(x)$$\nwhere:\n* $f_t(x)$ = Weak learner (tree) trained to fit pseudo-residuals (gradients) of step $t-1$\n* $\\eta$ = Learning rate (shrinkage)",
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
    analogy:
      "The network is a student being graded on confidence. Saying '90% sure it's a cat' when it's actually a dog gets heavily penalized. Saying '55% sure it's a cat' when it's a dog is bad, but not as catastrophic. Cross-entropy rewards accurate confidence, not just correct guesses.",
    math: "Binary Cross-Entropy Loss:\n$$\\mathcal{L} = -\\frac{1}{N} \\sum_{i=1}^{N} \\left[ y_i \\log(p_i) + (1 - y_i) \\log(1 - p_i) \\right]$$\nwhere:\n* $y_i$ = Ground truth label ($0$ or $1$)\n* $p_i$ = Predicted probability",
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
    analogy:
      "ReLU is a bouncer at a club. Positive signals get in; anything zero or below gets turned away. Without this gating, all the layers would just be one big linear equation and the network couldn't learn anything curved or complex. The bouncer creates the non-linearity that makes deep learning powerful.",
    math: "Rectified Linear Unit (ReLU):\n$$\\text{ReLU}(x) = \\max(0, x)$$\nOther activations include:\n* $\\text{Sigmoid}(x) = \\frac{1}{1 + e^{-x}}$\n* $\\tanh(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}$",
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
    analogy:
      "Imagine words as cities on a map, where meaning determines geography. 'King' and 'Queen' are in the same neighborhood. 'Happy' and 'Joyful' are next-door neighbors. 'Bank' (money) and 'Bank' (river) are on opposite sides of town. The embedding layer learns this map automatically from examples.",
    math: "Embedding Lookup:\n$$E_t = \\text{EmbeddingTable}[\\text{token\\_id}_t]$$\nAverage Pooling:\n$$v_{\\text{doc}} = \\frac{1}{T} \\sum_{t=1}^{T} E_t$$",
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
    analogy:
      "A surfer reads the ocean by watching the last several waves — their height, timing, and rhythm — to anticipate the next one. Time series forecasting does the same: the model studies the past W steps of a signal and learns which patterns reliably predict what comes next.",
    math: "Lookback Predictor:\n$$\\hat{y}_t = f(y_{t-1}, y_{t-2}, \\dots, y_{t-w})$$\nMean Absolute Error (MAE):\n$$\\text{MAE} = \\frac{1}{N} \\sum_{i=1}^{N} |y_i - \\hat{y}_i|$$",
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
    analogy:
      "Instead of replacing your camera lens to shoot in a new style, you snap a thin corrective filter onto the front. The original lens (pre-trained weights) stays untouched and continues working perfectly. You only train the tiny filter (ΔW) for your specific task. When you're done, peel it off and the camera is back to stock.",
    math: "LoRA Parameter Adaptation:\n$$W_{\\text{updated}} = W_{\\text{frozen}} + \\Delta W$$\n$$\\Delta W = \\frac{\\alpha}{r} (B A)$$\nwhere:\n* $B \\in \\mathbb{R}^{d \\times r}$ and $A \\in \\mathbb{R}^{r \\times k}$ are low-rank adapter matrices\n* $r \\ll \\min(d, k)$ is the rank\n* $\\alpha$ is a constant scaling factor",
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
    analogy:
      "The model is like a highlighter, not a writer. Given a paragraph and a question, it doesn't compose an answer — it reads every word and asks 'is this where the answer starts?' and 'is this where the answer ends?', then highlights the winning span. All the knowledge is already in the passage; the model just learns to point.",
    math: "Span Probability:\n$$P(\\text{start} = i) = \\frac{e^{S_i}}{\\sum_{k} e^{S_k}}, \\quad P(\\text{end} = j) = \\frac{e^{E_j}}{\\sum_{k} e^{E_k}}$$\nwhere:\n* $S_i$ = Start logit score for token $i$\n* $E_j$ = End logit score for token $j$",
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
    analogy:
      "Think of sliding a small flashlight beam across a dark photograph in a grid pattern. At each position, the flashlight (filter) checks if a specific pattern — an edge, a curve, a corner — is present. The brighter the response, the stronger the match. Stacking dozens of different flashlights builds a full description of what's in the image.",
    math: "2D Convolution:\n$$S(i, j) = \\sum_{m} \\sum_{n} I(i - m, j - n) K(m, n)$$\nwhere:\n* $I$ = Input image matrix\n* $K$ = Filter kernel matrix",
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
    analogy:
      "A GAN is a forger locked in an arms race with an art detective. The forger (Generator) gets better at faking Picassos; the detective (Discriminator) gets better at spotting fakes. Neither can rest — every improvement by one forces an upgrade from the other. Training ends when the forger is so good the detective can only guess randomly.",
    math: "Minimax Objective:\n$$\\min_{G} \\max_{D} V(D, G) = \\mathbb{E}_{x \\sim p_{\\text{data}}} [\\log D(x)] + \\mathbb{E}_{z \\sim p_{z}} [\\log (1 - D(G(z)))]$$",
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
    analogy:
      "Reading a novel with a sticky-note bookmark. As you turn pages, you add notes for important plot points and cross out ones that are no longer relevant. The LSTM's cell state is that bookmark — the forget gate decides what to cross out, the input gate decides what new note to add, and the output gate decides which notes to actually use right now.",
    math: "Cell State Update:\n$$C_t = f_t \\odot C_{t-1} + i_t \\odot \\tanh(W_c [h_{t-1}, x_t] + b_c)$$\nwhere:\n* $f_t$ = Forget gate activation vector\n* $i_t$ = Input gate activation vector\n* $\\odot$ = Element-wise (Hadamard) product",
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
    analogy:
      "A GPS coordinate stored to 8 decimal places is ultra-precise but takes up lots of space. Round it to 2 decimal places and you lose almost no navigational accuracy, but the file shrinks dramatically. Quantization does the same to neural network weights — trading a tiny sliver of precision for 4× smaller models and much faster inference.",
    math: "Quantization Mapping:\n$$q = \\text{round}\\left( \\frac{r}{S} \\right) + Z$$\nwhere:\n* $r$ = Real float weight/activation value\n* $S$ = Scale factor (positive float)\n* $Z$ = Zero-point (integer offset)",

    whatToObserve: [
      "Check the performance bar chart comparing FP32 vs. INT8 models.",
      "Observe that the INT8 quantized model size is ~4x smaller and latency is significantly lower, while accuracy remains almost identical.",
    ],
  },
  "numpy/transformer": {
    title: "Transformer",
    overview:
      "Transformers are the core architecture behind modern generative AI (like GPT, Claude, Gemini). This lesson implements a full causal Transformer block from scratch in NumPy, including positional encoding, multi-head causal self-attention, layer normalization, residual connections, and a feed-forward network, trained to predict the next character in a text sequence.",
    concepts: [
      {
        term: "Positional Encoding",
        definition:
          "Adds deterministic sinusoidal patterns to input embeddings to inject sequence order information into the model's representations.",
      },
      {
        term: "Causal Masking",
        definition:
          "A triangular mask applied to attention scores that prevents tokens from looking at future tokens, enabling autoregressive next-token prediction.",
      },
      {
        term: "Multi-Head Attention",
        definition:
          "Splits embedding vectors into multiple heads so the model can simultaneously focus on different features or token relationships.",
      },
      {
        term: "Layer Normalization",
        definition:
          "Normalizes activations across the feature dimension for each token, stabilizing and accelerating the training process.",
      },
      {
        term: "Residual Connections",
        definition:
          "Adds the input of a sub-layer to its output, allowing gradients to flow directly backward during backpropagation and preventing vanishing gradients.",
      },
    ],
    analogy:
      "Imagine writing a cooperative story where each writer sits in a row and can only see the words written before them. They read the history, focus on the most important plot points (multi-head attention), write a summary of the situation (feed-forward), and pass the paper forward. Autoregressive generation means the model repeats this step-by-step to write entire sentences.",
    math: "Positional Encoding Sinusoids:\n$$PE_{(pos, 2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)$$\n$$PE_{(pos, 2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)$$\n\nMulti-Head Causal Attention:\n$$\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, ..., \\text{head}_h) W^O$$\n$$\\text{head}_i = \\text{softmax}\\left(\\frac{Q_i K_i^T}{\\sqrt{d_k}} + M\\right) V_i$$\nwhere $M$ is a causal mask where $M_{ij} = 0$ for $i \\ge j$, and $M_{ij} = -\\infty$ for $i < j$.",
    whatToObserve: [
      "Observe the strictly lower-triangular shape of the attention alignment matrix heatmaps, reflecting that tokens cannot attend to future tokens due to causal masking.",
      "Watch the autoregressive text generation progress: at early epochs, the generated text is gibberish, but it converges to the target quote as training completes.",
      "Notice how the training loss decreases and character-level accuracy increases as the model learns to reconstruct the sequence.",
    ],
  },
  "pytorch/rag": {
    title: "Retrieval-Augmented Generation (RAG)",
    overview:
      "RAG is a technique that combines external document search with a generative large language model. By retrieving relevant facts from a local vector database and inserting them into the prompt, RAG grounds the generator in truth, preventing hallucinations and allowing LLMs to answer questions about private or real-time data.",
    concepts: [
      {
        term: "Semantic Embeddings",
        definition:
          "High-dimensional dense vector representations of text capturing conceptual meaning, generated by passing text through an encoder model.",
      },
      {
        term: "Cosine Similarity",
        definition:
          "A metric measuring the similarity in direction between two vectors, used to find semantically related documents for a query.",
      },
      {
        term: "Vector Database",
        definition:
          "A specialized storage and search system optimized for indexing and retrieving high-dimensional vectors by nearest-neighbor distance.",
      },
      {
        term: "Hallucination",
        definition:
          "The tendency of LLMs to generate plausible-sounding but completely incorrect or fabricated facts when they lack correct context.",
      },
      {
        term: "Prompt Augmentation",
        definition:
          "Grounding the generator by embedding retrieved document contexts directly into the input prompt before text generation.",
      },
    ],
    analogy:
      "Think of taking an open-book exam. Instead of trying to memorize every fact in the world (which leads to guessing and making things up), you read the question, search your textbook for the relevant page (retrieval), read the information on that page (augmentation), and write down a precise answer based on what you read (generation).",
    math: "Cosine Similarity:\n$$\\text{Similarity}(u, v) = \\frac{u \\cdot v}{\\|u\\| \\|v\\|} = \\frac{\\sum u_i v_i}{\\sqrt{\\sum u_i^2} \\sqrt{\\sum v_i^2}}$$\n\nRetrieval-Augmented Prompt Template:\n$$\\text{Prompt} = \\text{Context} \\cup \\text{Query}$$\n$$\\text{Context} = \\{D_i \\mid i \\in \\text{top-K}(\\text{Similarity}(E(Q), E(D)))\\}$$",
    whatToObserve: [
      "Compare the generated answers with and without retrieval: the baseline model without context makes up details (hallucinates), while the RAG model with context is accurate.",
      "Observe the 2D visualization of the embedding space: semantically related documents and the query are clustered close to each other, with similarity values labeled.",
      "Notice the cosine similarity bar chart, showing which documents in the corpus match the user's selected query.",
    ],
  },
};
