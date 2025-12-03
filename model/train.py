# train.py
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Callbacks
callbacks = [
    # Save best model
    keras.callbacks.ModelCheckpoint(
        'models/kana_best.h5',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    ),
    
    # Early stopping
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    
    # Reduce learning rate on plateau
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001
    ),
    
    # TensorBoard logging
    keras.callbacks.TensorBoard(
        log_dir='logs',
        histogram_freq=1
    )
]