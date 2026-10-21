import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
import os

# Create directory to save plots
os.makedirs("plots", exist_ok=True)

class MLP(nn.Module):
    def __init__(self, activation_fn):
        super(MLP, self).__init__()
        # ---------------------------------------------------------
        # TODO: Define a 3-layer MLP (Input: 2, Hidden1: 16, Hidden2: 16, Output: 1)
        # Use the 'activation_fn' passed in the arguments for hidden layers.
        # Use nn.Sigmoid() for the final output layer to get probabilities.
        # Example: self.fc1 = nn.Linear(2, 16)
        # ---------------------------------------------------------
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)
        self.activation = activation_fn
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        # ---------------------------------------------------------
        # TODO: Implement the forward pass sequentially through your layers
        # ---------------------------------------------------------
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.output_activation(self.fc3(x))

# Helper function to plot the decision boundary (Do not modify)
def plot_decision_boundary(model, X, y, title, ax):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        preds = model(grid).round().numpy()
        
    Z = preds.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax.scatter(X[:, 0], X[:, 1], c=y.numpy().flatten(), s=10, cmap=plt.cm.RdYlBu, edgecolors='k')
    ax.set_title(title)

def run():
    # 1. Generate Non-linear Dataset (Concentric Circles)
    X_np, y_np = make_circles(n_samples=1000, noise=0.05, factor=0.5, random_state=42)
    X = torch.FloatTensor(X_np)
    y = torch.FloatTensor(y_np).view(-1, 1)

    epochs = 1000
    learning_rate = 0.05
    metrics = {}
    loss_results = {}
    trained_models = {}

    activations = {
        "Sigmoid": nn.Sigmoid(),
        "ReLU": nn.ReLU()
    }

    for name, act_fn in activations.items():
        print(f"Training MLP with {name} activation...")
        # Reset seed so both networks start with identical linear parameters.
        torch.manual_seed(42)
        model = MLP(act_fn)
        
        # ---------------------------------------------------------
        # TODO: Define the loss function (BCE) and optimizer (SGD)
        # ---------------------------------------------------------
        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate) 
        
        losses = []
        for epoch in range(epochs):
            # ---------------------------------------------------------
            # TODO: Implement the standard PyTorch training loop
            # 1. Set model to train mode: model.train()
            # 2. Forward pass: calculate predictions
            # 3. Calculate loss
            # 4. Zero the gradients: optimizer.zero_grad()
            # 5. Backward pass: loss.backward()
            # 6. Update weights: optimizer.step()
            # ---------------------------------------------------------
            
            model.train()
            predictions = model(X)
            loss = criterion(predictions, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_loss = loss.item() 
            losses.append(current_loss)

        model.eval()
        with torch.no_grad():
            final_predictions = model(X)
            final_loss = criterion(final_predictions, y).item()
            accuracy = (final_predictions.round() == y).float().mean().item()
        print(f"{name}: initial BCE={losses[0]:.6f}, final BCE={final_loss:.6f}, "
              f"training accuracy={accuracy:.2%}")
        metrics[name] = {"initial_loss": losses[0], "final_loss": final_loss,
                         "training_accuracy": accuracy, "epochs": epochs,
                         "seed": 42, "learning_rate": learning_rate}
        loss_results[name] = losses
        trained_models[name] = model

    # ---------------------------------------------------------
    # Advanced Comprehensive Visualization (Do not modify)
    # ---------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot 1: Original Data
    axs[0, 0].scatter(X_np[:, 0], X_np[:, 1], c=y_np, s=10, cmap=plt.cm.RdYlBu, edgecolors='k')
    axs[0, 0].set_title("Original Dataset (Concentric Circles)")
    
    # Subplot 2: Loss Comparison
    for name, losses in loss_results.items():
        axs[0, 1].plot(losses, label=name)
    axs[0, 1].set_title("Training Loss: Sigmoid vs ReLU")
    axs[0, 1].set_xlabel("Epochs")
    axs[0, 1].set_ylabel("Loss (BCE)")
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Subplot 3: Sigmoid Decision Boundary
    plot_decision_boundary(trained_models["Sigmoid"], X, y, "Sigmoid Decision Boundary", axs[1, 0])
    
    # Subplot 4: ReLU Decision Boundary
    plot_decision_boundary(trained_models["ReLU"], X, y, "ReLU Decision Boundary", axs[1, 1])
    
    plt.tight_layout()
    save_path = "plots/activation_comprehensive_analysis.png"
    plt.savefig(save_path)
    print(f"Comprehensive plot saved to {save_path}")
    plt.close(fig)
    return metrics

if __name__ == "__main__":
    run()