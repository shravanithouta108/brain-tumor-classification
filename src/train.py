import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Sequential as KerasSequential
from sklearn.model_selection import train_test_split

from config import (
    IMAGE_SIZE, CHANNELS, NUM_CLASSES,
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    VALIDATION_SPLIT, MODEL_DIR, TRAIN_DIR, TEST_DIR
)
from preprocessing import load_dataset


def build_model():
    """
    Builds a simple CNN suitable for a mini project.
    """
    model = keras.Sequential([
        # Data augmentation layers (only active during training, automatically OFF at prediction time)
        layers.RandomFlip("horizontal", input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
        layers.RandomRotation(0.03),
        layers.RandomZoom(0.05),
        
        # First convolution block
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # Second convolution block
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # Third convolution block
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        # Flatten feature maps into a 1D list
        layers.Flatten(),

        # Dense (fully connected) layer for decision-making
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),  # helps prevent overfitting

        # Output layer: one neuron per class, softmax gives probabilities
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    print("Loading and preparing data...")
    X_train_full, y_train_full = load_dataset(TRAIN_DIR)

    # Split training data into actual-training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=VALIDATION_SPLIT,
        random_state=42,
        stratify=y_train_full
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")

    model = build_model()
    model.summary()

    print("\nStarting training...\n")
    early_stop = EarlyStopping(
        monitor="val_loss",       
        patience=5,                
        restore_best_weights=True 
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stop]
    )
    # Save the trained model
    import os
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "brain_tumor_cnn.keras")
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    # Save training history for later graphing (Stage 8)
    import json
    history_path = os.path.join(MODEL_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history.history, f)
    print(f"Training history saved to: {history_path}")
