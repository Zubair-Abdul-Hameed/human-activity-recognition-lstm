# import torch
# from sklearn.metrics import classification_report

# def evaluate(model, test_loader, device):
#     model.eval()

#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for X, y in test_loader:
#             X, y = X.to(device), y.to(device)

#             outputs = model(X)
#             preds = torch.argmax(outputs, dim=1)

#             correct += (preds == y).sum().item()
#             total += y.size(0)

#     acc = correct / total
#     print("Test Accuracy:", acc)

#     return acc

import torch
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def evaluate(model, test_loader, device):
    model.eval()

    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for X, y in test_loader:

            X = X.to(device)
            y = y.to(device)

            outputs = model(X)

            preds = torch.argmax(outputs, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = correct / total

    print("\n==============================")
    print(f"Test Accuracy: {acc:.4f}")
    print("==============================")

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            digits=4
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            all_labels,
            all_preds
        )
    )

    return acc