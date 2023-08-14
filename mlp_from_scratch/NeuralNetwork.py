import numpy as np
from FullyConnectedLayer import FullyConnectedLayer, FullyConnectedLayerWithScale, QFullyConnectedLayerWithScale
from ConvLayer import ConvLayer, CustomMaxPool, CustomFlatten, QConvLayer
from Activations import *
from quantizer import quantize, quantize_po2
import os
from tensorflow import keras
from keras import layers
from qkeras import QActivation
from qkeras import quantizers


class NeuralNetwork:
    """ vanilla NN """
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.layers = []
        self.layers.append(FullyConnectedLayer(input_size, 256))
        self.layers.append(ReLU())
        self.layers.append(FullyConnectedLayer(256, 256))
        self.layers.append(ReLU())
        self.layers.append(FullyConnectedLayer(256, output_size))

        self.softmax = Softmax()
    

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output


    def backward(self, grad_output, learning_rate):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate)


    def train(self, inputs, targets, learning_rate, num_epochs, batch_size=None):
        for epoch in range(num_epochs):
            loss = 0.0
            for batch_inputs, y_true in self.get_batches(inputs, targets, batch_size):
                
                # Forward pass
                z = self.forward(batch_inputs)

                # apply softmax
                y_pred = self.softmax.forward(z)

                # Compute loss
                loss += self.cross_entropy_loss_with_logits(y_pred, y_true)
                
                # Compute the derivative of the loss
                dz = self.cross_entropy_loss_with_logits_derivative(y_pred, y_true)
                
                # backward pass
                self.backward(dz, learning_rate)

            loss /= len(inputs)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss}")


    def predict(self, inputs):
        outputs = []
        for input in inputs:
            output = self.forward(input)
            predicted_class = np.argmax(output)
            outputs.append(predicted_class)
        return np.array(outputs)


    def cross_entropy_loss_with_logits(self, output, targets):
        num_samples = output.shape[0]
        loss = np.sum(-targets * np.log(output + 1e-8)) / num_samples
        return loss


    def cross_entropy_loss_with_logits_derivative(self, output, targets):
        # https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
        # The output of the network must pass to the softmax to this function here works
        # the derivative is quite complex and involves a lot of tricks.
        grad_output = output - targets
        return grad_output


    def get_batches(self, inputs, targets, batch_size=None):
        if batch_size is None or batch_size >= len(inputs):
            yield inputs, targets
        else:
            num_batches = len(inputs) // batch_size
            for i in range(num_batches):
                start = i * batch_size
                end = (i + 1) * batch_size
                yield inputs[start:end], targets[start:end]
            if len(inputs) % batch_size != 0:
                yield inputs[num_batches * batch_size:], targets[num_batches * batch_size:]



class NeuralNetworkWithScale:
    """ rede neural com tratamento de escala de pesos e ativações """

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.layers = []
        self.layers.append(FullyConnectedLayerWithScale(input_size, 256))
        self.layers.append(ReLU())
        self.layers.append(FullyConnectedLayerWithScale(256, 256))
        self.layers.append(ReLU())
        self.layers.append(FullyConnectedLayerWithScale(256, output_size))

        self.softmax = Softmax()
    

    def forward(self, inputs):
        # descobre a escala do dado de entrada
        x_scale = np.max(np.abs(inputs))    
        
        # escala entrada e atribui a variavel output que entrará no laço
        output = inputs / x_scale          

        for layer in self.layers:

            if isinstance(layer, FullyConnectedLayerWithScale):
                output = layer.foward_with_scale(output, x_scale=x_scale)
                x_scale = layer.output_scale

            else:
                output = layer.forward(output)

        # desnormaliza saída
        output = output * x_scale

        return output


    def backward(self, grad_output, learning_rate):

        # faz essa multiplicação para padronizar operações de retropropagação nas camadas
        grad_output = grad_output * self.layers[-1].output_scale

        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate)


    def train(self, inputs, targets, learning_rate, num_epochs, batch_size=None):
        for epoch in range(num_epochs):
            loss = 0.0
            for batch_inputs, y_true in self.get_batches(inputs, targets, batch_size):
                
                # Forward pass
                z = self.forward(batch_inputs)                
                
                # apply softmax
                y_pred = self.softmax.forward(z)

                # Compute loss
                loss += self.cross_entropy_loss_with_logits(y_pred, y_true)
                
                # Compute the derivative of the loss
                dz = self.cross_entropy_loss_with_logits_derivative(y_pred, y_true)
                
                # backward pass
                self.backward(dz, learning_rate)

            loss /= len(inputs)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss}")


    def predict(self, inputs):
        outputs = []
        for input in inputs:
            output = self.forward(input)
            predicted_class = np.argmax(output)
            outputs.append(predicted_class)
        return np.array(outputs)


    def cross_entropy_loss_with_logits(self, output, targets):
        num_samples = output.shape[0]
        loss = np.sum(-targets * np.log(output + 1e-8)) / num_samples
        return loss


    def cross_entropy_loss_with_logits_derivative(self, output, targets):
        # https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
        # The output of the network must pass to the softmax to this function here works
        # the derivative is quite complex and involves a lot of tricks.
        grad_output = output - targets
        return grad_output


    def get_batches(self, inputs, targets, batch_size=None):
        if batch_size is None or batch_size >= len(inputs):
            yield inputs, targets
        else:
            num_batches = len(inputs) // batch_size
            for i in range(num_batches):
                start = i * batch_size
                end = (i + 1) * batch_size
                yield inputs[start:end], targets[start:end]
            if len(inputs) % batch_size != 0:
                yield inputs[num_batches * batch_size:], targets[num_batches * batch_size:]                




class QNeuralNetworkWithScale:
    """ rede neural com tratamento de escala e quantização de pesos e ativações """

    def __init__(self, input_size, output_size):
        # iterations of dnn training
        self.iteration = 0        

        # input and output size of the NN
        self.input_size = input_size
        self.output_size = output_size

        # loss function grad output scale
        self.grad_output_scale = tf.constant(1., tf.float32)
        self.input_scale = tf.constant(1., tf.float32)

        self.layers = []
        self.layers.append(QFullyConnectedLayerWithScale(input_size, 256))
        self.layers.append(QReLU())
        self.layers.append(QFullyConnectedLayerWithScale(256, 256))
        self.layers.append(QReLU())
        self.layers.append(QFullyConnectedLayerWithScale(256, output_size, is_output_layer=True))

        self.softmax = Softmax()

        for l in self.layers:
            if isinstance(l, QFullyConnectedLayerWithScale):
                print(l.input_size, l.output_size)


        # accuracy history
        self.acc_hist = []
        # loss history
        self.loss_hist = []


    def forward(self, inputs):
        # descobre a escala do dado de entrada
        cp_inputs = inputs
        xs = tf.reduce_max(tf.abs(cp_inputs))    
        self.input_scale = 0.9 * self.input_scale + 0.1 * xs
        xs = self.input_scale
        
        # escala entrada e atribui a variavel output que entrará no laço ### TODO: testar fazer média móvel de entradas, mas deve piorar ACC
        output = cp_inputs / quantize_po2(self.input_scale)

        # quantiza a entrada...
        output = quantize(output, True)

        for layer in self.layers:

            if isinstance(layer, QFullyConnectedLayerWithScale):
                output = layer.qforward(output, xs=xs)
                xs = layer.output_scale

            elif isinstance(layer, QReLU):
                output = layer.forward(output)

            else:
                print("não identificado!")

        # desescala saída
        output = output * xs
        self.layers[-1].output_scale = tf.constant(1., tf.float32) # como a saída da última camada é escalada -> quantizada -> desescalada, logo a escala da saída é 1...

        return output


    def backward(self, grad_output, learning_rate):
        # clip grad
        grad_output = tf.clip_by_value(grad_output, -2, 2)        

        # escala gradiente com média móvel
        self.grad_output_scale = 0.99 * self.grad_output_scale + 0.01 * tf.reduce_max(tf.abs(grad_output))
        grad_output_scale = self.grad_output_scale
        grad_output /= quantize_po2(grad_output_scale)

        # quantiza o gradiente
        grad_output = quantize(grad_output, True, False)

        for layer in reversed(self.layers):
            if isinstance(layer, QFullyConnectedLayerWithScale):
                grad_output = layer.backward_with_scale(grad_output, grad_output_scale, learning_rate)
                grad_output_scale = layer.grad_output_scale
            elif isinstance(layer, QReLU):
                grad_output = layer.backward(grad_output, learning_rate)
            else:
                print("pau!")



    def train(self, inputs, targets, learning_rate, num_epochs, batch_size=None, x_val = None, y_val = None):
        # zerout num of iteration        
        self.iteration = 0

        for epoch in range(num_epochs):
            loss = 0.0
            for batch_inputs, y_true in self.get_batches(inputs, targets, batch_size):
                # iteration step
                self.iteration += 1

                # Forward pass
                z = self.forward(batch_inputs)                
                
                # apply softmax
                y_pred = self.softmax.forward(z)

                # Compute loss
                loss += self.cross_entropy_loss_with_logits(y_pred, y_true)
                
                # Compute the derivative of the loss
                dz = self.cross_entropy_loss_with_logits_derivative(y_pred, y_true)
                
                # backward pass
                self.backward(dz, learning_rate)                
                

            loss /= len(inputs)
            self.loss_hist.append(loss)

            str_train_log = f"Epoch {epoch+1}/{num_epochs}, Loss: {loss} "
            if x_val is not None and y_val is not None:
                # validation
                z = self.forward(x_val)
                y_pred = tf.argmax(z, axis=-1)

                # Calculate accuracy
                accuracy = tf.reduce_mean(tf.cast(y_pred == tf.argmax(y_val, axis=-1), tf.float32))
                self.acc_hist.append(accuracy)

                str_train_log += f"Accuracy: {accuracy * 100}%"
                
            print(str_train_log)


    def predict(self, inputs, batch_size=None):

        if batch_size is None:
            outputs = []
            for input in inputs:
                output = self.forward(input)
                predicted_class = tf.argmax(output)
                outputs.append(predicted_class)        
            return tf.stack(outputs)
        else:            
            outputs = []
            for batch_inputs in self.get_batches(inputs, batch_size=batch_size):        
                output = self.forward(batch_inputs)
                predicted_class = tf.argmax(output, axis=-1)
                outputs.append(predicted_class)
            outputs = tf.concat(outputs, axis=0)
            return outputs
                

    def cross_entropy_loss_with_logits(self, output, targets):
        num_samples = output.shape[0]
        loss = tf.reduce_sum(-targets * tf.math.log(output + 1e-8)) / num_samples
        return loss


    def cross_entropy_loss_with_logits_derivative(self, output, targets):
        # https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
        # The output of the network must pass to the softmax to this function here works
        # the derivative is quite complex and involves a lot of tricks.
        grad_output = output - targets
        return grad_output


    def get_batches(self, inputs, targets=None, batch_size=None):
        if batch_size is None or batch_size >= len(inputs):
            yield inputs, targets
        else:
            if targets is None:
                num_batches = len(inputs) // batch_size
                for i in range(num_batches):
                    start = i * batch_size
                    end = (i + 1) * batch_size
                    yield inputs[start:end]
                if len(inputs) % batch_size != 0:
                    yield inputs[num_batches * batch_size:]

            else:
                num_batches = len(inputs) // batch_size
                for i in range(num_batches):
                    start = i * batch_size
                    end = (i + 1) * batch_size
                    yield inputs[start:end], targets[start:end]
                if len(inputs) % batch_size != 0:
                    yield inputs[num_batches * batch_size:], targets[num_batches * batch_size:]                


    def save_weights(self, path):
        """ recieves a path where all model variable will be saved """

        if not os.path.exists(path=path):
            os.makedirs(path)

        # para cada camada
        for i, l in enumerate(self.layers):
            if isinstance(l, QFullyConnectedLayerWithScale):
                # salva pesos
                np.save(f"{path}/layer.{i}.weights", l.qw.numpy())

                # salva bias
                np.save(f"{path}/layer.{i}.bias", l.qb.numpy())

                # salva escalas
                np.save(f"{path}/layer.{i}.weights_scale", l.weights_scale.numpy())
                np.save(f"{path}/layer.{i}.input_scale", l.input_scale.numpy())
                np.save(f"{path}/layer.{i}.output_scale", l.output_scale.numpy())
                np.save(f"{path}/layer.{i}.grad_weights_scale", l.grad_weights_scale.numpy())
                np.save(f"{path}/layer.{i}.grad_bias_scale", l.grad_bias_scale.numpy())
                np.save(f"{path}/layer.{i}.grad_output_scale", l.grad_output_scale.numpy())

        
        # para a rede
        np.save(f"{path}/net.grad_output_scale",  self.grad_output_scale.numpy())
                       


    def load_weights(self, path):
        """ recieves a path from where all model variable will be loaded """

        # para cada camada
        for i, l in enumerate(self.layers):
            if isinstance(l, QFullyConnectedLayerWithScale):
                # salva pesos
                l.qw = tf.constant(np.load(f"{path}/layer.{i}.weights.npy"))
    
                # salva bias
                l.qb = tf.constant(np.load(f"{path}/layer.{i}.bias.npy"))

                # salva escalas
                l.weights_scale = tf.constant(np.load(f"{path}/layer.{i}.weights_scale.npy"))
                l.input_scale = tf.constant(np.load(f"{path}/layer.{i}.input_scale.npy"))
                l.output_scale = tf.constant(np.load(f"{path}/layer.{i}.output_scale.npy"))
                l.grad_weights_scale = tf.constant(np.load(f"{path}/layer.{i}.grad_weights_scale.npy"))
                l.grad_bias_scale = tf.constant(np.load(f"{path}/layer.{i}.grad_bias_scale.npy"))
                l.grad_output_scale = tf.constant(np.load(f"{path}/layer.{i}.grad_output_scale.npy"))

        
        # para a rede
        self.grad_output_scale = tf.constant(np.load(f"{path}/net.grad_output_scale.npy"))



    def load_layers_from_model(self, mlp):
        """ recieves a model from where all model variable will be loaded 
            also, quantize the weights
        """

        # limpa array de camadas
        self.layers.clear()

        # find last layer
        last_fc_layer_idx = 0
        for i, l in enumerate(mlp.layers):
            if isinstance(l, keras.layers.Dense): 
                last_fc_layer_idx = i


        # para cada camada
        for i, l in enumerate(mlp.layers):
            if isinstance(l, keras.layers.Dense):        
                # print("instantiating weights from ", l.name)
                qfc = QFullyConnectedLayerWithScale(l.weights[0].shape[0],l.weights[0].shape[1])
                

                # pega os pesos em fp32
                fpw = l.weights[0].numpy()        
                fpb  = np.reshape(l.weights[1].numpy(), (1, -1))
                
                # get scale
                w_scale = np.max(np.abs(fpw))
                
                # do scaling
                fpw_scaled = fpw / w_scale
                fpb_scaled = fpb / w_scale
                
                # quantiza pesos escalados
                qw = quantize(fpw_scaled, True, False)
                    
                # atribui o peso quantizado e escala
                qfc.qw = qw
                qfc.weights_scale = w_scale
                
                # quantiza e atribui bias escalados
                qb = quantize(fpb_scaled, True, False)
                qfc.qb = qb

                if last_fc_layer_idx == i:
                    qfc.is_output_layer = True

                # salva layer
                self.layers.append(qfc)


            if isinstance(l, keras.layers.ReLU) or isinstance(l, QActivation):          
                # print("instantiating relu")      
                self.layers.append(QReLU())

        # print("loaded layers", self.layers)


class LeNet:
    """ vanilla LeNet NN """
    def __init__(self, input_shape, output_size):
        self.batch_size = input_shape[0]
        self.input_shape = input_shape
        self.output_size = output_size

        self.layers = []
        self.layers.append(ConvLayer(nfilters=16, kernel_size=3, input_channels=input_shape[-1], strides=[1,1,1,1], padding='SAME'))
        self.layers.append(ReLU())
        self.layers.append(CustomMaxPool(ksize=2, stride=(2,2)))
        self.layers.append(ConvLayer(nfilters=32, kernel_size=3, input_channels=input_shape[-1], strides=[1,1,1,1], padding='SAME'))
        self.layers.append(ReLU())
        self.layers.append(CustomMaxPool(ksize=2, stride=(2,2)))
        self.layers.append(CustomFlatten(input_shape=[7, 7, 32]))
        self.layers.append(FullyConnectedLayer(1568, 256))
        self.layers.append(ReLU())
        self.layers.append(FullyConnectedLayer(256, 256))
        self.layers.append(ReLU())
        self.layers.append(FullyConnectedLayer(256, output_size))
        self.softmax = Softmax()
    

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output


    def backward(self, grad_output, learning_rate):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate)


    def train(self, inputs, targets, learning_rate, num_epochs,  x_val=None, y_val=None):
        for epoch in range(num_epochs):
            loss = 0.0
            for batch_inputs, y_true in self.get_batches(inputs, targets, self.batch_size):
                
                # Forward pass
                z = self.forward(batch_inputs)                

                # apply softmax
                y_pred = self.softmax.forward(z)

                # Compute loss
                loss += self.cross_entropy_loss_with_logits(y_pred, y_true)
                
                # Compute the derivative of the loss
                dz = self.cross_entropy_loss_with_logits_derivative(y_pred, y_true)
                
                # backward pass
                self.backward(dz, learning_rate)

            loss /= len(inputs)
            
            str_train_log = f"Epoch {epoch+1}/{num_epochs}, Loss: {loss} "
            if x_val is not None and y_val is not None:
                # validation
                z = self.forward(x_val)
                y_pred = tf.argmax(z, axis=-1, output_type=tf.int32)

                # Calculate accuracy
                accuracy = tf.reduce_mean(tf.cast(y_pred == tf.argmax(y_val, axis=-1, output_type=tf.int32), tf.float32)).numpy()              

                str_train_log += f"Accuracy: {accuracy * 100}%"
                
            print(str_train_log)



    def predict(self, inputs, batch_size=None):

        if batch_size is None:
            outputs = []
            for input in inputs:
                output = self.forward(input)
                predicted_class = tf.argmax(output)
                outputs.append(predicted_class)        
            return tf.stack(outputs, axis=0)
        else:            
            outputs = []
            for batch_inputs in self.get_batches(inputs, batch_size=batch_size):        
                output = self.forward(batch_inputs)
                predicted_class = tf.argmax(output, axis=-1)
                outputs.append(predicted_class)
            outputs = tf.concat(outputs, axis=0)
            return outputs






    def cross_entropy_loss_with_logits(self, output, targets):
        num_samples = output.shape[0]                
        loss = tf.reduce_sum(-targets * tf.cast(tf.math.log(output + 1e-16), tf.float32)) / num_samples
        return loss


    def cross_entropy_loss_with_logits_derivative(self, output, targets):
        # https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
        # The output of the network must pass to the softmax to this function here works
        # the derivative is quite complex and involves a lot of tricks.        
        grad_output = tf.subtract(output, targets)
        return grad_output


    def get_batches(self, inputs, targets=None, batch_size=None):
        if batch_size is None or batch_size >= len(inputs):
            yield inputs, targets
        else:
            if targets is None:
                num_batches = len(inputs) // batch_size
                for i in range(num_batches):
                    start = i * batch_size
                    end = (i + 1) * batch_size
                    yield inputs[start:end]
                if len(inputs) % batch_size != 0:
                    yield inputs[num_batches * batch_size:]

            else:
                num_batches = len(inputs) // batch_size
                for i in range(num_batches):
                    start = i * batch_size
                    end = (i + 1) * batch_size
                    yield inputs[start:end], targets[start:end]
                if len(inputs) % batch_size != 0:
                    yield inputs[num_batches * batch_size:], targets[num_batches * batch_size:]         



class QLeNet:
    """ Quantized LeNet-like NN """
    def __init__(self, input_shape, output_size, batch_size):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.output_size = output_size

        self.layers = []
        self.layers.append(QConvLayer(nfilters=16, kernel_size=3, input_channels=input_shape[-1], strides=[1,1,1,1], padding='SAME'))
        self.layers.append(QReLU())
        self.layers.append(CustomMaxPool(ksize=2, stride=(2,2)))
        self.layers.append(QConvLayer(nfilters=32, kernel_size=3, input_channels=input_shape[-1], strides=[1,1,1,1], padding='SAME'))
        self.layers.append(QReLU())
        self.layers.append(CustomMaxPool(ksize=2, stride=(2,2)))
        self.layers.append(CustomFlatten(input_shape=[7, 7, 32]))
        self.layers.append(QFullyConnectedLayerWithScale(1568, 256))
        self.layers.append(QReLU())
        self.layers.append(QFullyConnectedLayerWithScale(256, 256))
        self.layers.append(QReLU())
        self.layers.append(QFullyConnectedLayerWithScale(256, output_size, is_output_layer=True))

        self.softmax = Softmax()

        # loss function grad output scale
        self.grad_output_scale = 1

        # input image scale
        self.input_scale  = tf.constant(1., tf.float32)
        self.is_hist = []

        self.freeze_conv = False


    def load_layers_from_model(self, lenet, from_layer=0):
        
        # clear previously created layers
        self.layers.clear()

        # find last layer
        last_fc_layer_idx = 0
        for i, l in enumerate(lenet.layers):
            if isinstance(l, keras.layers.Dense): 
                last_fc_layer_idx = i
        
        # variavel para setar todas camadas apos flatten como training = True
        training = True

        for i, l in enumerate(lenet.layers[from_layer:]):            
            if isinstance(l, keras.layers.Conv2D):                    
                # print("instanciating conv layer...", l.output_shape)
                l.weights[0].shape[0],l.weights[0].shape[1]

                w_shape = l.weights[0].shape
                nfilters = w_shape[3]
                kernel_size = w_shape[0]
                input_channels = w_shape[2]
                strides=[1,1,1,1] ### TODO: variable strides
                padding= l.padding.upper()
                
                # create QCONVLAYER
                qfc = QConvLayer(nfilters, kernel_size, input_channels, strides, padding)
                
                fpw = l.weights[0].numpy()        
                fpb  = l.weights[1].numpy()
                
                w_scale = np.max(np.abs(fpw))
                
                fpw_scaled = fpw / w_scale
                qw = quantize(fpw_scaled, True, True)
                
                # atribui o peso quantizado
                qfc.qw = qw
                qfc.weights_scale = w_scale                                                    
                fpb_scaled = fpb / w_scale
                qb = quantize(fpb_scaled, True, False)
                qfc.qb = qb
                
                # não sera treinado CONV no CIFAR10
                qfc.training = True
                # training = False

                self.layers.append(qfc)


            if isinstance(l, keras.layers.MaxPool2D):    
                # print("instanciating MaxPool2D...", l.output_shape)
                dn_maxpool = CustomMaxPool(l.pool_size, l.strides, l.padding.upper())
                self.layers.append(dn_maxpool)

            if isinstance(l, keras.layers.Flatten):    
                # print("instanciating Flatten...", l.output_shape)
                self.layers.append(CustomFlatten(l.input_shape[1:])) # without batch
                training = True # treinar após flatten

            if isinstance(l, keras.layers.Dense):        
                # print("instanciating Dense...", l.output_shape)

                qfc = QFullyConnectedLayerWithScale(l.weights[0].shape[0],l.weights[0].shape[1])
                
                fpw = l.weights[0].numpy()        
                fpb  = l.weights[1].numpy()
                
                w_scale = np.max(np.abs(fpw))
                
                fpw_scaled = fpw / w_scale
                qw = quantize(fpw_scaled, True, True)
                
                # atribui o peso quantizado
                qfc.qw = qw
                qfc.weights_scale = w_scale
                    
                
                fpb_scaled = fpb / w_scale
                qb = quantize(fpb_scaled, True, False)
                qfc.qb = qb

                if i == last_fc_layer_idx:
                    qfc.is_output_layer = True

                self.layers.append(qfc)


            if isinstance(l, keras.layers.ReLU) or isinstance(l, QActivation):
                if isinstance(l, QActivation):
                    if isinstance(l.quantizer, quantizers.quantized_relu_po2):                
                        self.layers.append(QReLU(training))
                else:
                    self.layers.append(QReLU(training))

    def restart_fc_layers(self):
        for l in self.layers:
            if isinstance(l, QFullyConnectedLayerWithScale):
                l.reset_weights()

    
    def forward(self, inputs):
        # descobre a escala do dado de entrada
        x = inputs
        xs = tf.reduce_max(tf.abs(x))
        self.input_scale = self.input_scale * 0.9 + 0.1 * xs
        self.is_hist.append(self.input_scale)
        xs = self.input_scale
        # escala entrada e atribui a variavel output que entrará no laço
        output = x / quantize_po2(self.input_scale)

        # quantiza a entrada
        output = quantize(output, True, True)

        for layer in self.layers:
            if isinstance(layer, QConvLayer):
                # print("QConvLayer")
                output = layer.qforward(output, xs=xs)
                xs = layer.output_scale
            elif isinstance(layer, QFullyConnectedLayerWithScale):
                # print("QFullyConnectedLayerWithScale")
                output = layer.qforward(output, xs=xs)
                xs = layer.output_scale
            elif isinstance(layer, QReLU):
                # print("QReLU")
                output = layer.forward(output)            
            elif isinstance(layer, CustomFlatten):
                # print("CustomFlatten")
                output = layer.forward(output)            
            elif isinstance(layer, CustomMaxPool):
                # print("CustomMaxPool")
                output = layer.forward(output)
            else:
                print("não identificado!")

        # desescala saída
        output = output * xs
        # como a saída da última camada é escalada -> quantizada -> desescalada, logo a escala da saída é 1...
        self.layers[-1].output_scale = tf.constant(1, tf.float32) 
        

        return output
    

    def backward(self, grad_output, learning_rate):
        # clip grad
        grad_output = tf.clip_by_value(grad_output, -2, 2)        

        # escala gradiente com média móvel
        self.grad_output_scale = 0.99 * self.grad_output_scale + 0.01 * tf.reduce_max(tf.abs(grad_output))
        grad_output_scale = self.grad_output_scale
        grad_output /= quantize_po2(grad_output_scale)

        # quantiza o gradiente
        grad_output = quantize(grad_output, True, False)

        for i, layer in enumerate(reversed(self.layers)):
            # print(f"processing back prop of layer {len(self.layers) -i}... {type(layer)} ...")
            if isinstance(layer, QConvLayer):                
                grad_output = layer.qbackward(grad_output, grad_output_scale, learning_rate)
                grad_output_scale = layer.grad_output_scale
            elif isinstance(layer, QFullyConnectedLayerWithScale):
                grad_output = layer.backward_with_scale(grad_output, grad_output_scale, learning_rate)
                grad_output_scale = layer.grad_output_scale
            elif isinstance(layer, QReLU):
                grad_output = layer.backward(grad_output, learning_rate)
            elif isinstance(layer, CustomFlatten):                
                if not self.freeze_conv: # interrompe backprop apos flatten
                    grad_output = layer.backward(grad_output, learning_rate)
                else:
                    return 0
            elif isinstance(layer, CustomMaxPool):
                grad_output = layer.backward(grad_output, learning_rate)
            else:
                print("pau!")


    def train(self, inputs, targets, learning_rate, num_epochs,  x_val=None, y_val=None):
        for epoch in range(num_epochs):
            loss = 0.0
            for batch_inputs, y_true in self.get_batches(inputs, targets, self.batch_size):
                
                # Forward pass
                z = self.forward(batch_inputs)                                

                # apply softmax
                y_pred = self.softmax.forward(z)

                # Compute loss
                loss += self.cross_entropy_loss_with_logits(y_pred, y_true)
                
                # Compute the derivative of the loss
                dz = self.cross_entropy_loss_with_logits_derivative(y_pred, y_true)
                
                # backward pass
                self.backward(dz, learning_rate)

            loss /= len(inputs)
            
            str_train_log = f"Epoch {epoch+1}/{num_epochs}, Loss: {loss} "
            if x_val is not None and y_val is not None:
                # validation
                z = self.forward(x_val)
                y_pred = tf.argmax(z, axis=-1, output_type=tf.int32)

                # Calculate accuracy
                accuracy = tf.reduce_mean(tf.cast(y_pred == tf.argmax(y_val, axis=-1, output_type=tf.int32), tf.float32)).numpy()              

                str_train_log += f"Accuracy: {accuracy * 100}%"
                
            print(str_train_log)



    def predict(self, inputs, batch_size=None, apply_argmax=True):

        if batch_size is None:
            outputs = []
            for input in inputs:
                output = self.forward(input)
                predicted_class = tf.argmax(output)
                outputs.append(predicted_class)        
            return tf.stack(outputs, axis=0)
        else:            
            outputs = []
            for batch_inputs in self.get_batches(inputs, batch_size=batch_size):        
                output = self.forward(batch_inputs)
                if apply_argmax:
                    predicted_class = tf.argmax(output, axis=-1)
                    outputs.append(predicted_class)
                else:
                    outputs.append(output)
            outputs = tf.concat(outputs, axis=0)
            return outputs


    def cross_entropy_loss_with_logits(self, output, targets):
        num_samples = output.shape[0]                   
        loss = tf.reduce_sum(-targets * tf.cast(tf.math.log(output + 1e-16), tf.float32)) / num_samples
        return loss


    def cross_entropy_loss_with_logits_derivative(self, output, targets):
        # https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
        # The output of the network must pass to the softmax to this function here works
        # the derivative is quite complex and involves a lot of tricks.        
        grad_output = tf.subtract(output, targets)
        return grad_output


    def get_batches(self, inputs, targets=None, batch_size=None):
        if batch_size is None or batch_size >= len(inputs):
            yield inputs, targets
        else:
            if targets is None:
                num_batches = len(inputs) // batch_size
                for i in range(num_batches):
                    start = i * batch_size
                    end = (i + 1) * batch_size
                    yield inputs[start:end]
                if len(inputs) % batch_size != 0:
                    yield inputs[num_batches * batch_size:]

            else:
                num_batches = len(inputs) // batch_size
                for i in range(num_batches):
                    start = i * batch_size
                    end = (i + 1) * batch_size
                    yield inputs[start:end], targets[start:end]
                if len(inputs) % batch_size != 0:
                    yield inputs[num_batches * batch_size:], targets[num_batches * batch_size:]         
