import tensorflow as tf
from keras import layers, models

def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x

    # First convolution
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Second convolution
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Shortcut connection
    if stride > 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Add the shortcut to the output
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)

    return x

def resnet(input_shape, num_classes, num_blocks=[2, 2, 2, 2]):
    input_layer = layers.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv2D(64, 7, strides=2, padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Residual blocks
    filters = 64
    for i, num_blocks in enumerate(num_blocks):
        for j in range(num_blocks):
            stride = 2 if j == 0 and i != 0 else 1
            x = residual_block(x, filters, stride=stride)
        filters *= 2

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected layer
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=x)

    return model

# Example usage:
input_shape = (224, 224, 3)  # Example input shape for RGB images
num_classes = 1000  # Example number of classes for ImageNet

model = resnet(input_shape, num_classes)
model.summary()