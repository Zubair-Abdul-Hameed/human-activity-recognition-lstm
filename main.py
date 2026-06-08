import torch
from src.data_loader import load_dataset
from src.dataset import HAR_Dataset
from src.model import LSTM_HAR
from src.train import train_model
from src.evaluate import evaluate
from torch.utils.data import DataLoader
from src.preprocessing import StandardScaler3D
from src.plot_results import save_training_plots



def main():

    # ---------------- LOAD DATA ----------------
    X_train, y_train, X_test, y_test = load_dataset()

    # ---------------- PREPROCESSING ----------------
    scaler = StandardScaler3D()

    X_train = scaler.fit_transform(X_train)


    X_test = scaler.transform(X_test)

    # ---------------- DATASETS ----------------
    train_ds = HAR_Dataset(X_train, y_train)
    test_ds = HAR_Dataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    # ---------------- MODEL ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM_HAR()

    # ---------------- TRAIN ----------------

    model, history = train_model(
        model,
        train_loader,
        test_loader,
        device,
    )

    save_training_plots(history)
    # ---------------- TEST ----------------
    evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()
