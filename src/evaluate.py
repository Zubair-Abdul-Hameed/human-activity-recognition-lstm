import torch


def evaluate(model, test_loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)

            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print("Test Accuracy:", acc)

    return acc