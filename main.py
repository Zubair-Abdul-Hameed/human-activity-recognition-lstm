import torch
from src.data_loader import load_dataset
from src.dataset import HAR_Dataset
from src.model import LSTM_HAR
from src.train import train_model
from src.evaluate import evaluate
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def main():

    # ---------------- LOAD DATA ----------------
    X_train, y_train, X_test, y_test = load_dataset()

    # ---------------- TRAIN/VAL SPLIT ----------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    # ---------------- DATASETS ----------------
    train_ds = HAR_Dataset(X_train, y_train)
    val_ds = HAR_Dataset(X_val, y_val)
    test_ds = HAR_Dataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)

    # ---------------- MODEL ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM_HAR()

    # ---------------- TRAIN ----------------
    model = train_model(model, train_loader, val_loader, device)

    # ---------------- TEST ----------------
    evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()