# train.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

def train_model(model, X, y, epochs=20, batch_size=64):
    print("\n📊 Splitting data...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    print(f"  Train: {len(X_train)}")
    print(f"  Validation: {len(X_val)}")
    print(f"  Test: {len(X_test)}")
    
    y_train_cat = tf.keras.utils.to_categorical(y_train, 2)
    y_val_cat = tf.keras.utils.to_categorical(y_val, 2)
    y_test_cat = tf.keras.utils.to_categorical(y_test, 2)
    
    model.compile(
        optimizer=Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    class_weight = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weight))
    print(f"  Class weights: {class_weight_dict}")
    
    callbacks = [
        ModelCheckpoint('models/best_model.h5', save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy')
    ]
    
    print("\n🚀 Training...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n📊 Evaluating...")
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Loss: {test_loss:.4f}")
    
    return history, test_acc