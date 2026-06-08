import numpy as np
import os

BASE_PATH = "data/UCI HAR Dataset"


def load_signals(split="train"):
    path = os.path.join(BASE_PATH, split, "Inertial Signals")

    acc_x = np.loadtxt(os.path.join(path, f"body_acc_x_{split}.txt"))
    acc_y = np.loadtxt(os.path.join(path, f"body_acc_y_{split}.txt"))
    acc_z = np.loadtxt(os.path.join(path, f"body_acc_z_{split}.txt"))

    gyro_x = np.loadtxt(os.path.join(path, f"body_gyro_x_{split}.txt"))
    gyro_y = np.loadtxt(os.path.join(path, f"body_gyro_y_{split}.txt"))
    gyro_z = np.loadtxt(os.path.join(path, f"body_gyro_z_{split}.txt"))
    total_acc_x = np.loadtxt(os.path.join(path, f"total_acc_x_{split}.txt"))
    total_acc_y = np.loadtxt(os.path.join(path, f"total_acc_y_{split}.txt"))
    total_acc_z = np.loadtxt(os.path.join(path, f"total_acc_z_{split}.txt"))

    X = np.stack([acc_x, acc_y, acc_z,
                  gyro_x, gyro_y, gyro_z, 
                  total_acc_x, total_acc_y, total_acc_z], axis=-1)

    return X


def load_labels(split="train"):
    path = os.path.join(BASE_PATH, split, f"y_{split}.txt")
    y = np.loadtxt(path).astype(int) - 1
    return y


def load_dataset():
    X_train = load_signals("train")
    y_train = load_labels("train")

    X_test = load_signals("test")
    y_test = load_labels("test")

    return X_train, y_train, X_test, y_test