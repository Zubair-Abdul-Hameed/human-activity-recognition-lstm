import os
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def evaluate(model, test_loader, device):
    os.makedirs("results", exist_ok=True)
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
            digits=4,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            all_labels,
            all_preds
        )
    )
    report = classification_report(
        all_labels,
        all_preds,
        digits=4
    )
    with open("results/classification_report.txt", "w") as f:
        f.write(report)
    cm = confusion_matrix(
        all_labels,
        all_preds
    )
    with open("results/confusion_matrix.txt", "w") as f:
        f.write(str(cm))

    return acc