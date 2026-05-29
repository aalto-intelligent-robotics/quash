import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np

class FastSVM(nn.Module):
    def __init__(self, n_samples=None, initial_gamma=1.0, C=1.0, learn_gamma=True, scale_gamma=False):
        super(FastSVM, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # GAMMA
        self.learn_gamma = learn_gamma
        self.scale_gamma = scale_gamma
        if self.learn_gamma:
            self.gamma = nn.Parameter(torch.tensor(initial_gamma))
        else:
            self.gamma = initial_gamma
        # C
        self.C = C
        # ALPHA
        if n_samples is not None:
            self.alpha = nn.Parameter(torch.zeros(n_samples))  # Initialize alphas to zero
        else:
            self.alpha = None
        # Bias
        self.bias = nn.Parameter(torch.zeros(1))  # Initialize bias to zero

    def rbf_kernel(self, X, Y):
        X = torch.as_tensor(X, dtype=torch.float32)
        Y = torch.as_tensor(Y, dtype=torch.float32)
        pairwise_sq_dists = torch.cdist(X, Y, p=2) ** 2
        return torch.exp(-self.gamma * pairwise_sq_dists)

    def forward(self, X, support_vectors, support_alphas, support_labels):
        """
        Compute the decision function for the input data X based on the support vectors,
        their corresponding alphas (Lagrange multipliers), and labels.

        Args:
            X (torch.Tensor): The input data for which predictions are made (e.g., X_test).
            support_vectors (torch.Tensor): The support vectors obtained from training.
            support_alphas (torch.Tensor): The learned Lagrange multipliers (alphas).
            support_labels (torch.Tensor): The labels associated with the support vectors.

        Returns:
            torch.Tensor: The decision function output for X.
        """
        # Compute the RBF kernel between the input data X and the support vectors
        K = self.rbf_kernel(X, support_vectors)

        # Weighted sum of support vectors' alphas and labels
        # `weighted_support` is the contribution of each support vector to the decision
        weighted_support = support_alphas * support_labels

        # Compute the decision function as a dot product of the kernel matrix and the weighted support
        decision = torch.matmul(K, weighted_support) + self.bias

        # Return the decision function output, which will later be used to make predictions
        return decision


    class HingeLoss(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, decision, labels):
            loss = torch.clamp(1 - labels * decision, min=0)
            return loss.mean()

    def fit(self, X_train, y_train, epochs=1, lr=0.01, batch_size=64):
        if self.alpha is None:
            n_samples = X_train.size(0)
            self.alpha = nn.Parameter(torch.zeros(n_samples))

        if self.scale_gamma:
            self.gamma = 1 / (X_train.shape[0])

        self.to(self.device)

        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        criterion = self.HingeLoss()  # Hinge loss with optional regularization

        dataset = data.TensorDataset(X_train, y_train)
        loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                decision = self(batch_X, X_train.to(self.device), self.alpha.to(self.device), y_train.to(self.device))
                loss = criterion(decision, batch_y)

                # Backward pass and optimization
                loss.backward()

                # Update alphas and parameters
                optimizer.step()

                # Manually clamp alphas to [0, C]
                self.alpha.data = torch.clamp(self.alpha.data, min=0, max=self.C)

                # Enforce sum(alpha_i * y_i) = 0 for constraint
                y_train  = y_train.to(self.device)
                self.alpha.data -= (self.alpha.data * y_train.to(self.device)).sum() * y_train / y_train.size(0)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

        self.support_vectors = X_train
        self.support_labels = y_train


    def predict(self, X_test, batch_size=64, support_vectors=None, support_alphas=None, support_labels=None):
        if support_vectors is None:
            support_vectors = self.support_vectors
        if support_labels is None:
            support_labels = self.support_labels
        if support_alphas is None:
            support_alphas = self.alpha

        self.to(self.device)
        X_test = X_test.to(self.device)
        dataset = data.TensorDataset(X_test)
        loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_predictions = []

        with torch.no_grad():  # Disable gradient computation for prediction
            for batch_X in loader:
                batch_X = batch_X[0].to(self.device)
                decision = self(batch_X, support_vectors.to(self.device), support_alphas.to(self.device), support_labels.to(self.device))
                predictions = torch.sign(decision).cpu()  # Move to CPU to save GPU memory
                all_predictions.append(predictions)

        return torch.cat(all_predictions).detach().cpu().numpy()