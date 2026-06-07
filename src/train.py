import torch
import torch.nn as nn


# ---------------- EARLY STOPPING ----------------
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return True  # improvement happened
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


# ---------------- TRAIN FUNCTION ----------------
def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    early_stopping = EarlyStopping(patience=5)

    best_model_path = "models/best_model.pt"

    model.to(device)

    for epoch in range(epochs):

        # ================= TRAIN =================
        model.train()
        train_loss = 0.0
        train_correct = 0
        total = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            outputs = model(X)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss (weighted for proper averaging)
            train_loss += loss.item() * X.size(0)

            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss /= total
        train_acc = train_correct / total


        # ================= VALIDATION =================
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total = 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)

                outputs = model(X)
                loss = criterion(outputs, y)

                val_loss += loss.item() * X.size(0)

                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == y).sum().item()
                total += y.size(0)

        val_loss /= total
        val_acc = val_correct / total


        # ================= LOGGING =================
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:2d} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_acc*100:.1f}% | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_acc*100:.1f}%"
        )

        print(f"  current lr: {current_lr:.6f}")


        # ================= EARLY STOPPING + SAVE BEST MODEL =================
        is_best = early_stopping(val_loss)

        if is_best:
            torch.save(model.state_dict(), best_model_path)
            print("  ✓ Best model saved")


        if early_stopping.early_stop:
            print("Early stopping triggered")
            break


    # ================= LOAD BEST MODEL BEFORE RETURN =================
    model.load_state_dict(torch.load(best_model_path))
    return model