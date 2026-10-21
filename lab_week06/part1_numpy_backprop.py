import numpy as np

def sigmoid(x):
    # ---------------------------------------------------------
    # TODO: Implement the sigmoid activation function
    # Formula: f(x) = 1 / (1 + exp(-x))
    # ---------------------------------------------------------
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    # ---------------------------------------------------------
    # TODO: Implement the derivative of the sigmoid function
    # Hint: It can be expressed using the sigmoid function itself: f(x) * (1 - f(x))
    # ---------------------------------------------------------
    value = sigmoid(x)
    return value * (1.0 - value)

def forward_backward(X, Y, W1, b1, W2, b2):
    """Return predictions, mean half-squared loss and analytic gradients."""
    z1 = X @ W1 + b1
    h = sigmoid(z1)
    z2 = h @ W2 + b2
    y_pred = sigmoid(z2)
    loss = 0.5 * np.mean((y_pred - Y) ** 2)
    delta2 = (y_pred - Y) / Y.size * sigmoid_derivative(z2)
    dW2 = h.T @ delta2
    db2 = delta2.sum(axis=0, keepdims=True)
    delta1 = (delta2 @ W2.T) * sigmoid_derivative(z1)
    dW1 = X.T @ delta1
    db1 = delta1.sum(axis=0, keepdims=True)
    return y_pred, float(loss), (dW1, db1, dW2, db2)


def run(iterations=5, hidden_size=2, seed=42, learning_rate=0.5,
        verbose=True, stop_at_tolerance=False):
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    # Initialize weights and biases deterministically for reproducible debugging
    rng = np.random.RandomState(seed)
    input_size = 2
    output_size = 1
    
    W1 = rng.uniform(-1, 1, (input_size, hidden_size))
    b1 = np.zeros((1, hidden_size))
    W2 = rng.uniform(-1, 1, (hidden_size, output_size))
    b2 = np.zeros((1, output_size))

    # Start with 5 for the basic requirement. Increase later for exploration!
    first_close = None
    losses = []

    if verbose:
        print(f"{'Iter':<5} | {'Loss':<8} | {'Output (y_pred)':<30} | {'Target':<15} | {'W2 Mean':<10}")
        print("-" * 80)

    for i in range(iterations):
        # ---------------------------------------------------------
        # TODO: 1. FORWARD PASS
        # Calculate hidden layer output (h) and final prediction (y_pred)
        # Hint: h = sigmoid(X.dot(W1) + b1)
        # ---------------------------------------------------------
        y_pred, loss, gradients = forward_backward(X, Y, W1, b1, W2, b2) 
        
        # Calculate Loss (Mean Squared Error)
        # loss = 0.5 * np.mean((y_pred - Y) ** 2)
        losses.append(loss)
        
        # ---------------------------------------------------------
        # TODO: 2. BACKWARD PASS (Chain Rule)
        # Calculate gradients: dW2, db2, dW1, db1
        # Error = 0.5 * (y_pred - Y)^2
        # Start with: d_Error/d_y_pred = (y_pred - Y) / number_of_samples
        # Then multiply by the derivative of the Sigmoid function.
        # ---------------------------------------------------------
        dW1, db1, dW2, db2 = gradients

        # ---------------------------------------------------------
        # TODO: 3. WEIGHT UPDATE
        # Update W1, b1, W2, b2 using the learning rate
        # Example: W2 -= learning_rate * dW2
        # ---------------------------------------------------------
        
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        # Assess predictions AFTER exactly i+1 parameter updates.
        final_pred = sigmoid(sigmoid(X @ W1 + b1) @ W2 + b2)
        if first_close is None and np.max(np.abs(final_pred - Y)) <= 0.1:
            first_close = i + 1

        # Print tabular results for the report
        if verbose:
            pred_str = ", ".join([f"{val[0]:.4f}" for val in y_pred])
            target_str = ", ".join([str(val[0]) for val in Y])
            w2_mean = np.mean(W2)
            print(f"{i+1:<5} | {loss:<8.4f} | [{pred_str}] | [{target_str}] | {w2_mean:<10.4f}")

        if first_close is not None and stop_at_tolerance:
            break

    # ---------------------------------------------------------
    # Output final parameters after all iterations
    # ---------------------------------------------------------
    if verbose:
        print("\n" + "=" * 40)
        print("FINAL MODEL PARAMETERS")
        print("=" * 40)
        print(f"W1 (Input -> Hidden):\n{W1}\n")
        print(f"b1 (Hidden Biases):\n{b1}\n")
        print(f"W2 (Hidden -> Output):\n{W2}\n")
        print(f"b2 (Output Biases):\n{b2}")
        print("=" * 40)
    return {"updates": i + 1, "first_close": first_close,
            "predictions": final_pred.ravel().tolist(),
            "losses": losses, "final_loss": float(0.5 * np.mean((final_pred - Y)**2)),
            "seed": seed, "hidden_size": hidden_size, "learning_rate": learning_rate}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--quiet", action="store_true", help="Print a JSON summary instead of iteration tables")
    parser.add_argument("--stop-at-tolerance", action="store_true", help="Stop when all absolute XOR errors are <= 0.1")
    args = parser.parse_args()
    if args.iterations < 1 or args.hidden_size < 1 or args.learning_rate <= 0:
        parser.error("iterations, hidden size and learning rate must be positive")
    result = run(args.iterations, args.hidden_size, args.seed, args.learning_rate,
                 verbose=not args.quiet, stop_at_tolerance=args.stop_at_tolerance)
    if args.quiet:
        import json
        result.pop("losses")
        print(json.dumps(result, indent=2))
