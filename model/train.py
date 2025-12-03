# train.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Import model from cnn_model.py
from cnn_model import create_kana_cnn


def plot_training_history(history, timestamp):
    """
    Plot training and validation accuracy/loss
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'logs/training_history_{timestamp}.png', dpi=150, bbox_inches='tight')
    print(f"✓ Training plots saved to logs/training_history_{timestamp}.png")
    plt.close()


def train_model(epochs=50, batch_size=32, learning_rate=0.001):
    """
    Train the kana recognition model
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
    """
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Load data
    print("Loading dataset...")
    data = np.load('../data/processed/kana_train_val_test.npz')
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"\nDataset loaded:")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Val:   {len(X_val):,} samples")
    print(f"  Test:  {len(X_test):,} samples")
    
    # Reshape for CNN (add channel dimension)
    X_train = X_train.reshape(-1, 64, 64, 1)
    X_val = X_val.reshape(-1, 64, 64, 1)
    X_test = X_test.reshape(-1, 64, 64, 1)
    
    # Convert labels to categorical
    y_train_cat = keras.utils.to_categorical(y_train, 92)
    y_val_cat = keras.utils.to_categorical(y_val, 92)
    y_test_cat = keras.utils.to_categorical(y_test, 92)
    
    print(f"\nInput shape: {X_train.shape}")
    print(f"Output shape: {y_train_cat.shape}")
    
    # Create model
    print("\nBuilding model...")
    model = create_kana_cnn(num_classes=92, input_shape=(64, 64, 1))
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel architecture:")
    model.summary()
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    callbacks = [
        # Save best model
        keras.callbacks.ModelCheckpoint(
            'models/kana_best.h5',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        ),
        
        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=f'logs/{timestamp}',
            histogram_freq=1
        ),
        
        # CSV logger
        keras.callbacks.CSVLogger(
            f'logs/training_{timestamp}.csv'
        )
    ]
    
    # Train
    print(f"\nTraining for {epochs} epochs with batch size {batch_size}...")
    print("="*60)
    
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*60)
    print("Training completed!")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    
    print(f"\nFinal Results:")
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy*100:.2f}%")
    
    # Plot training history
    plot_training_history(history, timestamp)
    
    # Save final model
    model.save(f'models/kana_final_{timestamp}.h5')
    print(f"\n✓ Final model saved to models/kana_final_{timestamp}.h5")
    print(f"✓ Best model saved to models/kana_best.h5")
    
    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Kana Recognition CNN')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("KANA RECOGNITION CNN - TRAINING")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print()
    
    model, history = train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    print("\n" + "="*60)
    print("TRAINING FINISHED!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Check logs/training_history_*.png for training curves")
    print("  2. Run: tensorboard --logdir=logs")
    print("  3. Evaluate: python evaluate.py")