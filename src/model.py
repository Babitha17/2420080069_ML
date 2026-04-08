# model.py
import tensorflow as tf
from tensorflow.keras import layers, models

def create_resnet34(input_shape=(187, 1), num_classes=2):
    def residual_block(x, filters, stride=1):
        shortcut = x
        x = layers.Conv1D(filters, 3, strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv1D(filters, 3, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.add([x, shortcut])
        x = layers.ReLU()(x)
        return x
    
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv1D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(3, strides=2, padding='same')(x)
    
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    
    x = residual_block(x, 512, stride=2)
    x = residual_block(x, 512)
    x = residual_block(x, 512)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)