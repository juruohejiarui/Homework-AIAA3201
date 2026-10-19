import numpy as np

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).
    This version is given to you as the reference: your vectorized
    version below must produce the same loss and gradient.

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_classes = W.shape[1]
    num_train = X.shape[0]

    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]  # gradient wrt W_j
                dW[:, y[i]] -= X[i]  # gradient wrt W_yi

    # Right now our loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    # Average and add regularization to the gradient
    dW /= num_train
    dW += 2 * reg * W

    return loss, dW

def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    num_train = X.shape[0]

    # ---------------------------------------------------------
    # TODO 2: Compute the scores for all samples at once
    # ---------------------------------------------------------
    # X has shape (N, D), W has shape (D, C). The score matrix has
    # shape (N, C), one row of class scores per sample.
    # Hint: a single matrix multiplication.
    scores = X.dot(W)

    # Extract correct class scores
    correct_class_scores = scores[np.arange(num_train), y]

    # Compute margins matrix
    margins = np.maximum(0, scores - correct_class_scores[:, np.newaxis] + 1)

    # Set correct class margins to 0
    margins[np.arange(num_train), y] = 0

    # Compute data loss
    data_loss = np.sum(margins) / num_train

    # Add regularization
    reg_loss = reg * np.sum(W * W)
    loss = data_loss + reg_loss

    # ---------------------------------------------------------
    # TODO 3: Implement vectorized gradient computation
    # ---------------------------------------------------------
    # Steps:
    # 1. Build a binary mask the same shape as margins, 1 where the
    #    margin is positive and 0 elsewhere.
    # 2. For each sample, count how many positive margins it has and
    #    write minus that count into its correct class position, so
    #    the correct class column collects the negative gradient.
    # 3. dW = X transposed times the mask, divided by num_train,
    #    plus the regularization term 2 * reg * W.
    binary = (margins > 0).astype(np.float64)
    binary[np.arange(num_train), y] = -np.sum(binary, axis=1)
    dW = X.T.dot(binary) / num_train
    dW += 2 * reg * W

    return loss, dW

class LinearClassifier:
    """
    A linear classifier with Stochastic Gradient Descent (SGD) training.
    """

    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape
        num_classes = np.max(y) + 1

        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            # ---------------------------------------------------------
            # TODO 4: Sample a minibatch of batch_size examples
            # ---------------------------------------------------------
            # Pick batch_size random indices with replacement and gather
            # the corresponding rows of X and y.
            # Hint: np.random.choice
            batch_indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            self.W -= learning_rate * grad

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        y_pred = np.zeros(X.shape[0])
        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis=1)
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns:
        A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        raise NotImplementedError

class LinearSVM(LinearClassifier):
    """
    A subclass that uses the Multiclass SVM loss function.
    """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)
