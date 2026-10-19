import pickle
import numpy as np
import os

def load_CIFAR_batch(filename):
    """Load single batch of CIFAR-10."""
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']

        # Convert to float32 and normalize to [0, 1]
        X = X.astype(np.float32) / 255.0

        # Convert labels to numpy array
        Y = np.array(Y)

        return X, Y

def load_CIFAR10(ROOT):
    """Load the entire CIFAR-10 dataset."""
    xs = []
    ys = []

    # Load 5 training batches
    for b in range(1, 6):
        f = os.path.join(ROOT, f'data_batch_{b}')
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)

    # Concatenate all training data
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)

    # Load test data
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))

    return Xtr, Ytr, Xte, Yte

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True):
    """
    Load CIFAR-10 dataset and perform preprocessing.
    Returns training set, validation set, and test set.

    Parameters:
    - num_training: number of training samples (default 49000)
    - num_validation: number of validation samples (default 1000)
    - num_test: number of test samples (default 1000)
    - subtract_mean: whether to subtract the mean image (default True)
    """
    # Load raw data
    cifar10_dir = 'cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Validate data sizes
    assert X_train.shape[0] == 50000, "Training data should have 50000 samples"
    assert X_test.shape[0] == 10000, "Test data should have 10000 samples"

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]

    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Reshape data to (N, D) format where N is number of samples, D is feature dimension
    # Original format: (N, 32, 32, 3) -> flatten to (N, 3072)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    return X_train, y_train, X_val, y_val, X_test, y_test

def preprocess_with_bias_trick(X):
    """
    Implement the bias trick: append a constant 1 to each sample vector.
    This allows us to merge the bias term b into the weight matrix W,
    enabling simplified computation f(x, W) = Wx without explicit bias terms.

    Parameters:
    - X: Input data with shape (N, D), where N is number of samples, D is feature dimension

    Returns:
    - X_bias: Data with bias trick applied, shape (N, D+1)
    """
    N, D = X.shape

    # ---------------------------------------------------------
    # TODO 1: Implement the bias trick
    # ---------------------------------------------------------
    # Append a constant 1 to the end of every sample vector so the
    # bias can be folded into W. The output shape is (N, D+1).
    # Hint: np.column_stack or np.hstack with a column of ones.
    ones = np.ones((N, 1), dtype=X.dtype)
    X_bias = np.hstack((X, ones))

    return X_bias
