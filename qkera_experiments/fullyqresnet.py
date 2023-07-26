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


def make_qlayer(x, planes, blocks, stride=1, name=None):
    downsample = None
    inplanes = x.shape[3]
    if stride != 1 or inplanes != planes:
        downsample = [
            QConv2D(filters=planes, kernel_size=1, strides=stride, use_bias=False, 
                    kernel_quantizer=quantized_po2(4,1,True) , 
                    name=f'{name}.0.downsample.0'),
             QBatchNormalization(name=f'{name}.0.downsample.1'),
        ]

    x = qbasic_block(x, planes, stride, downsample, name=f'{name}.0')
    for i in range(1, blocks):
        x = qbasic_block(x, planes, name=f'{name}.{i}')

    return x



def qresnet_cifar(x, blocks_per_layer, num_classes=100):
    x = layers.ZeroPadding2D(padding=1, name='conv1_pad')(x)

    x = QConv2D(filters=16, kernel_size=3, strides=1, use_bias=False, kernel_quantizer=quantized_po2(4,1, use_stochastic_rounding=True), name='qconv1')(x)    
    x = QBatchNormalization(name='bn1')(x) 
    x = QActivation(quantized_relu_po2(4, 1, 
                                       use_stochastic_rounding=True), name='qrelu1')(x) 
    
    ## layer 1
    x = make_qlayer(x, 16, blocks_per_layer[0], 1, "qlayer1")

    ## layer 2
    x = make_qlayer(x, 32, blocks_per_layer[1], 2, "qlayer2")

    ## layer 3
    x = make_qlayer(x, 64, blocks_per_layer[2], 2, "qlayer3")
    
    x = layers.GlobalAveragePooling2D(name='avgpool')(x)
    
    x = QDense(units=num_classes, kernel_quantizer=quantized_po2(4,1,use_stochastic_rounding=True),
               bias_quantizer=quantized_po2(4, 1, use_stochastic_rounding=True), name='fc')(x)

    return x


def resnet20(x, **kwargs):    
    return qresnet_cifar(x, [3, 3, 3], **kwargs)
        
def resnet32(x, **kwargs):    
    return qresnet_cifar(x, [5, 5, 5], **kwargs)

