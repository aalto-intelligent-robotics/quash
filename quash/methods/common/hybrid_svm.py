import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import sklearn

class HybridSVM(nn.Module):
    def __init__(self):
        super(HybridSVM, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clf = sklearn.svm.SVC()

    def rbf_kernel(self, X, Y):
        X = torch.as_tensor(X, dtype=torch.float32)
        Y = torch.as_tensor(Y, dtype=torch.float32)
        pairwise_sq_dists = torch.cdist(X, Y, p=2) ** 2
        return torch.exp(-self.gamma * pairwise_sq_dists)

    def forward(self, X, support_vectors, support_alphas, support_labels):
        # Kernel between inputs and support vectors
        K = self.rbf_kernel(X, support_vectors)
        weighted_support = support_alphas * support_labels
        decision = torch.matmul(K, weighted_support) + self.bias
        return decision

    class HingeLoss(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, decision, labels):
            loss = torch.clamp(1 - labels * decision, min=0)
            return loss.mean()

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)
        self.gamma = 1 / (X_train.shape[0] * X_train.var())
        self.support_vectors = X_train
        self.support_labels = y_train

    def predict(self, X_test, batch_size=64, support_vectors=None, support_labels=None):
        support_alphas = torch.tensor(self.clf.dual_coef_)
        self.bias = self.clf.intercept_
        if support_vectors is None:
            support_vectors = self.support_vectors
        if support_labels is None:
            support_labels = self.support_labels

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