import torch
from src.data_loader import load_dataset
from src.dataset import HAR_Dataset
from src.model import LSTM_HAR
from src.train import train_model
from src.evaluate import evaluate
from torch.utils.data import DataLoader
from src.preprocessing import StandardScaler3D
from sklearn.model_selection import train_test_split


def main():

    # ---------------- LOAD DATA ----------------
    X_train, y_train, X_test, y_test = load_dataset()

    #remove this code block later
    import numpy as np

    def check_class_distribution(y, name="dataset"):
        counts = np.bincount(y)

        print(f"\n===== {name.upper()} CLASS DISTRIBUTION =====")
        for i, c in enumerate(counts):
            print(f"Class {i}: {c} samples")
        print("=====================================\n")


    check_class_distribution(y_train, "train")
    check_class_distribution(y_test, "test")
    # ---------------- TRAIN/VAL SPLIT ----------------
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X_train,
    #     y_train,
    #     test_size=0.2,
    #     random_state=42,
    #     stratify=y_train
    # ) #comfirm if I can spilt the train set without worrying that the task of same subject will appear in both train and val spilt(i.e subject dataleakage, I think thats what its called)

    # ---------------- PREPROCESSING ----------------
    scaler = StandardScaler3D()

    X_train = scaler.fit_transform(X_train)

    # X_val = scaler.transform(X_val)

    X_test = scaler.transform(X_test)
    print("\n===== NORMALIZATION CHECK =====")

    print("Train Mean:", X_train.mean())
    print("Train Std :", X_train.std())

    print("Test Mean :", X_test.mean())
    print("Test Std  :", X_test.std())

    print("===============================\n")
    # ---------------- DATASETS ----------------
    train_ds = HAR_Dataset(X_train, y_train)
    # val_ds = HAR_Dataset(X_val, y_val)
    test_ds = HAR_Dataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)

    # ---------------- MODEL ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM_HAR()

    # ---------------- TRAIN ----------------
    model = train_model(model, train_loader, test_loader, device)

    # ---------------- TEST ----------------
    evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()