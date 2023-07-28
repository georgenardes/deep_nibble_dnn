from qkeras import *
import tensorflow as tf
from tensorflow import keras 
from keras import layers



def qconv3x3(x, out_planes, stride=1, name=None):
    x = layers.ZeroPadding2D(padding=1, name=f'{name}_pad')(x)
    return QConv2D(filters=out_planes, kernel_size=3, strides=stride,
                    use_bias=False, kernel_quantizer=quantized_po2(4,1,True), 
                    name=name)(x)


def qbasic_block(x, planes, stride=1, downsample=None, name=None):
    identity = x

    out = qconv3x3(x, planes, stride=stride, name=f'{name}.conv1')
    out = QBatchNormalization(name=f'{name}.bn1')(out)
    out = QActivation(quantized_relu_po2(4,1,use_stochastic_rounding=True), 
                      name=f'{name}.relu1')(out)

    out = qconv3x3(out, planes, name=f'{name}.conv2')
    out = QBatchNormalization(name=f'{name}.bn2')(out)

    if downsample is not None:
        for layer in downsample:
            identity = layer(identity)

    
    out = layers.Add(name=f'{name}.add')([identity, out])
    out = QActivation(quantized_relu_po2(4,1,use_stochastic_rounding=True),
                       name=f'{name}.relu2')(out)

    return out


    x = QDense(units=num_classes, kernel_quantizer=quantized_po2(4,1,use_stochastic_rounding=True), bias_quantizer=quantized_po2(4,1), name='fc')(x)

    return x




def QVGG_16(x, num_outputs=10, **kwargs):
    # Block 1
    x = QConv2D(filters=64, kernel_size=3, padding="same", kernel_quantizer=quantized_po2(4,1,True), bias_quantizer=quantized_po2(4,1,False))(x)
    x = QActivation(quantized_relu_po2(4,1,use_stochastic_rounding=True))(x)
    x = QConv2D(filters=64, kernel_size=3, padding="same", kernel_quantizer=quantized_po2(4,1,True), bias_quantizer=quantized_po2(4,1,False))(x)
    x = QActivation(quantized_relu_po2(4,1,use_stochastic_rounding=True))(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # Block 2
    x = QConv2D(filters=128, kernel_size=3, padding="same", kernel_quantizer=quantized_po2(4,1,True), bias_quantizer=quantized_po2(4,1,False))(x)
    x = QActivation(quantized_relu_po2(4,1,use_stochastic_rounding=True))(x)
    x = QConv2D(filters=128, kernel_size=3, padding="same", kernel_quantizer=quantized_po2(4,1,True), bias_quantizer=quantized_po2(4,1,False))(x)    
    x = QActivation(quantized_relu_po2(4,1,use_stochastic_rounding=True))(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # Block 3
    x = QConv2D(filters=256, kernel_size=3, padding="same", kernel_quantizer=quantized_po2(4,1,True), bias_quantizer=quantized_po2(4,1,False))(x)
    x = QActivation(quantized_relu_po2(4,1,use_stochastic_rounding=True))(x)
    x = QConv2D(filters=256, kernel_size=3, padding="same", kernel_quantizer=quantized_po2(4,1,True), bias_quantizer=quantized_po2(4,1,False))(x)
    x = QActivation(quantized_relu_po2(4,1,use_stochastic_rounding=True))(x)
    x = QConv2D(filters=256, kernel_size=3, padding="same", kernel_quantizer=quantized_po2(4,1,True), bias_quantizer=quantized_po2(4,1,False))(x)    
    x = QActivation(quantized_relu_po2(4,1,use_stochastic_rounding=True))(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # # Block 4
    # x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    # x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    # x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    # x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    # 
    # # Block 5
    # x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    # x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    # x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    # x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # Classification head
    x = tf.keras.layers.Flatten()(x)    
    x = QDense(units=256, kernel_quantizer=quantized_po2(4,1,use_stochastic_rounding=True), bias_quantizer=quantized_po2(4,1))(x)    
    x = QActivation(quantized_relu_po2(4,1,use_stochastic_rounding=True))(x)
    x = QDense(units=256, kernel_quantizer=quantized_po2(4,1,use_stochastic_rounding=True), bias_quantizer=quantized_po2(4,1))(x)    
    x = QActivation(quantized_relu_po2(4,1,use_stochastic_rounding=True))(x)
    x = QDense(units=num_outputs, kernel_quantizer=quantized_po2(4,1,use_stochastic_rounding=True), bias_quantizer=quantized_po2(4,1))(x)
    
    return x


def VGG_16(x, num_classes=10, **kwargs):
    # Block 1
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # Block 2
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # Block 3
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # # Block 4
    # x = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(x)
    # x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(x)
    # x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(x)
    # x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    # # Block 5
    # x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    # x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    # x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    # x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    # Classification head
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(num_classes)(x)  # Assuming 1000 classes for ImageNet

    return x

