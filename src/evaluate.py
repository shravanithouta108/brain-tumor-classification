import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from config import TEST_DIR, MODEL_DIR, OUTPUT_DIR, CLASS_NAMES
from preprocessing import load_dataset


def plot_training_history():
    history_path = os.path.join(MODEL_DIR, "training_history.json")
    with open(history_path, "r") as f:
        history = json.load(f)

    epochs_range = range(1, len(history["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy plot
    axes[0].plot(epochs_range, history["accuracy"], label="Training Accuracy")
    axes[0].plot(epochs_range, history["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title("Training vs Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # Loss plot
    axes[1].plot(epochs_range, history["loss"], label="Training Loss")
    axes[1].plot(epochs_range, history["val_loss"], label="Validation Loss")
    axes[1].set_title("Training vs Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.savefig(save_path)
    print(f"Training curves saved to: {save_path}")
    plt.show()


def evaluate_on_test_set():
    print("Loading test data...")
    X_test, y_test = load_dataset(TEST_DIR)

    print("Loading trained model...")
    model_path = os.path.join(MODEL_DIR, "brain_tumor_cnn.keras")
    model = keras.models.load_model(model_path)

    print("Running predictions on test set...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)  # pick the class with highest probability

    # Overall accuracy
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Detailed report: precision, recall, F1-score per class
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    print(report)

    # Save report to a text file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write(report)
    print(f"Report saved to: {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Test Set")

    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to: {cm_path}")
    plt.show()


if __name__ == "__main__":
    plot_training_history()
    evaluate_on_test_set()
