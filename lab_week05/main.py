#!/usr/bin/env python3
"""
Lab 5: Linear Classifier & Multiclass SVM - Main Script

Lab Objectives:
1. Understand the mapping between features and weights in linear classifiers, master the Bias Trick
2. Implement multiclass SVM loss function (Hinge Loss) and its analytical gradient computation
3. Master Numpy vectorized operations, eliminate inefficient for-loops
4. Implement Stochastic Gradient Descent based on mini-batch data
5. Perform hyperparameter tuning to select the best model
"""

import numpy as np
import time
from data_utils import get_CIFAR10_data, preprocess_with_bias_trick
from classifiers import svm_loss_naive, svm_loss_vectorized, LinearSVM
from hyperparameter_tuning import hyperparameter_tuning, print_results

def main():
    print("=" * 60)
    print("Lab 5: Linear Classifier & Multiclass SVM")
    print("=" * 60)

    # 1. Data preparation and preprocessing
    print("\n1. Loading and preprocessing CIFAR-10 data...")

    # Load CIFAR-10 data
    X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()

    print(f"Training set: {X_train.shape[0]} samples, feature dimension: {X_train.shape[1]}")
    print(f"Validation set: {X_val.shape[0]} samples, feature dimension: {X_val.shape[1]}")
    print(f"Test set: {X_test.shape[0]} samples, feature dimension: {X_test.shape[1]}")

    # Implement Bias Trick: append constant 1 to each sample vector
    print("\nApplying Bias Trick: append constant 1 to each sample vector...")
    X_train = preprocess_with_bias_trick(X_train)
    X_val = preprocess_with_bias_trick(X_val)
    X_test = preprocess_with_bias_trick(X_test)

    print(f"Feature dimension after Bias Trick: {X_train.shape[1]} (originally 3072 + 1)")

    # 2. Compare performance of Naive and Vectorized versions
    print("\n2. Comparing performance of Naive and Vectorized SVM loss functions...")

    # Initialize weight matrix for testing
    num_classes = 10
    W = np.random.randn(X_train.shape[1], num_classes) * 0.001

    # Select a small batch for testing
    num_test_samples = 500
    X_batch = X_train[:num_test_samples]
    y_batch = y_train[:num_test_samples]
    reg = 1e-5

    # Test Naive version
    print("Testing Naive version (using for-loops)...")
    start_time = time.time()
    loss_naive, grad_naive = svm_loss_naive(W, X_batch, y_batch, reg)
    naive_time = time.time() - start_time
    print(f"loss: {loss_naive:.6f}")
    print(f"time: {naive_time:.4f}")

    # Test Vectorized version
    print("\nTesting Vectorized version (Numpy vectorization)...")
    start_time = time.time()
    loss_vectorized, grad_vectorized = svm_loss_vectorized(W, X_batch, y_batch, reg)
    vectorized_time = time.time() - start_time
    print(f"loss: {loss_vectorized:.6f}")
    print(f"time: {vectorized_time:.4f}")

    # Check if results are consistent
    loss_diff = abs(loss_naive - loss_vectorized)
    grad_diff = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')

    print("\nResults comparison:")
    print(f"loss difference: {loss_diff:.6f}")
    print(f"grad difference: {grad_diff:.6f}")
    print(f"speedup factor: {naive_time/vectorized_time:.2f}x")

    if loss_diff < 1e-9 and grad_diff < 1e-9:
        print("✓ Naive and Vectorized versions produce consistent results!")
    else:
        print("✗ Naive and Vectorized versions produce inconsistent results!")

    # 3. Train using LinearSVM
    print("\n3. Training with LinearSVM...")

    # Use default hyperparameters for preliminary training
    svm = LinearSVM()
    loss_history = svm.train(X_train, y_train,
                           learning_rate=1e-7,
                           reg=5e-4,
                           num_iters=1500,
                           batch_size=200,
                           verbose=True)

    # Evaluate training results
    y_train_pred = svm.predict(X_train)
    train_acc = np.mean(y_train == y_train_pred)

    y_val_pred = svm.predict(X_val)
    val_acc = np.mean(y_val == y_val_pred)

    print("\nPreliminary training results:")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    # 4. Hyperparameter tuning
    print("\n4. Performing hyperparameter tuning (grid search)...")

    best_svm, best_val_acc, results = hyperparameter_tuning(X_train, y_train, X_val, y_val)

    print_results(results)

    # 5. Evaluate best model on test set
    print("\n5. Evaluating best model on test set...")

    y_test_pred = best_svm.predict(X_test)
    test_acc = np.mean(y_test == y_test_pred)

    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Validation accuracy: {best_val_acc:.4f}")
    print("\nLab completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()