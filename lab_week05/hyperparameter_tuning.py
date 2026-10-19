import numpy as np
from classifiers import LinearSVM

def hyperparameter_tuning(X_train, y_train, X_val, y_val):
    """
    Perform hyperparameter tuning using grid search.

    Inputs:
    - X_train: Training data with shape (N, D)
    - y_train: Training labels with shape (N,)
    - X_val: Validation data with shape (M, D)
    - y_val: Validation labels with shape (M,)

    Returns:
    - best_svm: Best performing SVM model on validation set
    - best_val_acc: Best validation accuracy
    - results: Dictionary of all hyperparameter combinations and their validation accuracies
    """

    # Preset ranges for learning rates and regularization strengths
    learning_rates = [1e-7, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    regularization_strengths = [1e-7, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

    # Dictionary to store results
    results = {}
    best_val_acc = -1
    best_svm = None

    # ---------------------------------------------------------
    # TODO 5: Perform grid search over learning_rates and
    # regularization_strengths
    # ---------------------------------------------------------
    # For every (lr, reg) pair:
    #   1. create a LinearSVM
    #   2. train it with num_iters=1500 and batch_size=200
    #   3. measure accuracy on the training and the validation set
    #   4. store both accuracies in results[(lr, reg)] as a tuple
    #   5. keep the model that reaches the highest validation accuracy
    # The two prints below each step are already provided, keep them
    # so you can follow the progress.
    for lr in learning_rates:
        for reg in regularization_strengths:
            print(f'Training SVM with lr={lr:.1e}, reg={reg:.1e}')
            svm = LinearSVM()
            svm.train(X_train, y_train, learning_rate=lr, reg=reg,
                      num_iters=1500, batch_size=200)
            y_train_pred = svm.predict(X_train)
            train_acc = np.mean(y_train == y_train_pred)
            y_val_pred = svm.predict(X_val)
            val_acc = np.mean(y_val == y_val_pred)
            results[(lr, reg)] = (train_acc, val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_svm = svm
            print(f'  train accuracy: {train_acc:.4f}, val accuracy: {val_acc:.4f}')

    # Print best results
    print("\n" + "="*50)
    print("Best hyperparameter combination:")
    for lr, reg in results:
        train_acc, val_acc = results[(lr, reg)]
        if val_acc == best_val_acc:
            print(f'lr={lr:.1e}, reg={reg:.1e}')
            print(f'Validation accuracy: {val_acc:.4f}')
            break

    return best_svm, best_val_acc, results

def print_results(results):
    """
    Print all hyperparameter tuning results.
    """
    print("\nResults for all hyperparameter combinations:")
    print("lr\t\treg\t\ttrain_acc\tval_acc")
    print("-" * 50)

    for lr, reg in sorted(results.keys()):
        train_acc, val_acc = results[(lr, reg)]
        print(f'{lr:.1e}\t{reg:.1e}\t{train_acc:.4f}\t\t{val_acc:.4f}')
