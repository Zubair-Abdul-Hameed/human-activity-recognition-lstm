import matplotlib.pyplot as plt
import os


def save_training_plots(history):

    os.makedirs("results", exist_ok=True)

    # Loss Plot
    plt.figure()

    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig("results/loss_curve.png")
    plt.close()

    # Accuracy Plot
    plt.figure()

    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig("results/accuracy_curve.png")
    plt.close()