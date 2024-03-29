import numpy as np
from quantizer import quantize, stochastic_rounding, quantize_po2
import tensorflow as tf



class FullyConnectedLayer:
    """ this is a vanilla FC layer """

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = tf.constant(np.random.randn(input_size, output_size) * np.sqrt(2/input_size), tf.float32)
        self.biases = tf.zeros((1, output_size))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = tf.matmul(inputs, self.weights) + self.biases
        return self.output

    def backward(self, grad_output, learning_rate):
        # gradient calculation
        grad_input = tf.matmul(grad_output, self.weights, transpose_b=True)
        grad_weights = tf.matmul(self.inputs, grad_output, transpose_a=True)
        grad_biases = tf.reduce_sum(grad_output, axis=0, keepdims=True)

        # weight update
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases
        return grad_input
    


class FullyConnectedLayerWithScale:
    """ This is a FC layer with scale """
    

    def __init__(self, input_size, output_size):
        
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2/input_size)
        self.biases = np.zeros((1, output_size))

        #################################################
        # escala inicial dos pesos
        self.weights_scale = np.max(np.abs(self.weights))
        self.input_scale = None
        self.output_scale = 1
        # escala da escala de gradiente
        self.grad_output_scale = 1
        #################################################


    def forward(self, inputs):
        """ Default forward """

        self.inputs = inputs        
        self.output = np.matmul(inputs, self.weights) + self.biases
        return self.output
    
    
    def foward_with_scale(self, inputs, x_scale):

        # salva escala de entrada
        self.input_scale = x_scale
        # descobre a escala dos pesos com base no valor máximo
        self.weights_scale = np.max(np.abs(self.weights))
        # escala os pesos 
        w = self.weights / self.weights_scale
        # escala os biases com base na escala dos pesos e escala das ativações
        b = self.biases / (self.weights_scale * self.input_scale)
        #################################################

        # faz matmul e desescala pesos e biases        
        self.inputs = inputs
        self.output = (np.matmul(inputs, w) + b) * (self.weights_scale * self.input_scale)
        
        # descobre escala da saída com base em uma média
        self.output_scale = 0.9 * self.output_scale + 0.1 * np.max(np.abs(self.output))      

        # escala saída
        self.output = self.output / self.output_scale
        #################################################

        return self.output


    def backward(self, grad_output, learning_rate):
        
        # scaling gradients        
        grad_output = (grad_output / self.output_scale) * (self.weights_scale * self.input_scale)
        
        # gradient calculation
        grad_input = np.matmul(grad_output, self.weights.T / self.weights_scale)
        grad_weights = np.matmul(self.inputs.T, grad_output) / self.weights_scale
        grad_biases = np.sum(grad_output, axis=0, keepdims=True) / (self.weights_scale * self.input_scale)

        # weight update
        self.weights -= learning_rate * grad_weights
        self.biases -=  learning_rate * grad_biases
        return grad_input

    
    def backward_with_scale(self, grad_output, grad_scale, learning_rate):
        """ grad_output é o erro que chega para esta camada """

        # scaling gradients        
        grad_output = (grad_output / self.output_scale) * (self.weights_scale * self.input_scale) * (grad_scale)
        
        # gradient calculation
        grad_input = np.matmul(grad_output, self.weights.T / self.weights_scale)

        # para simular a operação em hardware, será necessário salvar o grad_weights escalado e quantizado 
        grad_weights = np.matmul(self.inputs.T, grad_output) / self.weights_scale
        grad_biases = np.sum(grad_output, axis=0, keepdims=True) / (self.weights_scale * self.input_scale)

        # weight update
        self.weights -= learning_rate * grad_weights
        self.biases -=  learning_rate * grad_biases

        # scale de grad_output or grad_of_input
        self.grad_output_scale = np.max(np.abs(grad_output))
        grad_input = grad_input / self.grad_output_scale

        return grad_input




class QFullyConnectedLayerWithScale:
    """ This is a FC layer with scale and SX4 Quantization"""
    

    def __init__(self, input_size, output_size, is_output_layer=False):
        
        self.input_size = input_size
        self.output_size = output_size

        w = tf.constant(np.random.randn(input_size, output_size) * np.sqrt(2/input_size), tf.float32)
        self.weights_scale = tf.reduce_max(tf.abs(w))
        self.qw = quantize(w/self.weights_scale, True, True)
        
        b = tf.zeros((1, output_size))        
        self.qb = quantize(b, True, False) # quantized bias
        
        
        #################################################
        # escala inicial dos pesos
        self.ws_hist = []
        self.bs_hist = []
        self.input_scale = tf.constant(1, tf.float32)
        self.output_scale = tf.constant(1, tf.float32)
        self.os_hist = []
        
        # escala de gradiente
        self.grad_output_scale = tf.constant(1, tf.float32)
        self.gos_hist = []
        self.grad_output_hist = []

        # escala de gradiente dos pesos
        self.grad_weights_scale = tf.constant(1, tf.float32)
        self.gws_hist = []
        self.grad_bias_scale = tf.constant(1, tf.float32)
        self.gbs_hist = []
        #################################################
                
        self.is_output_layer = is_output_layer
    
    
    def qforward(self, inputs, xs):
        # salva entrada para backprop, (entrada já vem quantizada)
        self.inputs = inputs

        # salva escala de entrada
        self.input_scale = xs                            
        qxs = quantize_po2(self.input_scale)
        qws = quantize_po2(self.weights_scale)

        # faz matmul e desescala pesos e biases
        self.output = (tf.matmul(inputs, self.qw) + self.qb) * (qws * qxs)
        self.output = tf.clip_by_value(self.output, -512, 512)

        # descobre escala da saída com base em uma média
        self.output_scale = 0.99 * self.output_scale + 0.01 * tf.reduce_max(self.output) # removido abs pq depois vem RELU
        qos = quantize_po2(self.output_scale)
        
        self.os_hist.append(qos)

        # escala saída
        self.output = self.output / qos

        # quantiza saída
        if self.is_output_layer: # FP32 as output
            self.output = self.output # quantize(self.output, stochastic_round=True, stochastic_zero=False)
        else:
            self.output = quantize(self.output, stochastic_round=True, stochastic_zero=True)
        #################################################

        return self.output

    
    def backward_with_scale(self, grad_output, grad_scale, learning_rate):
        """ 
        Esta função faz propagação do erro para os pesos e para a entrada.

        grad_output: esse parâmetro é o Erro que vem da camada l+1. Ele vem quantizado para Deep Nibble.
        grad_scale: esse é a Escala usada para normalizar o Erro antes de quantiza-lo para Deep Nibble
        learning_rate: taxa de aprendizado
        """                

        qws = quantize_po2(self.weights_scale)
        qxs = quantize_po2(self.input_scale)
        qgos = quantize_po2(grad_scale)        
        qos = quantize_po2(self.output_scale)

        self.gos_hist.append(qgos) 

        # gradient calculation. self.qw é a matriz de pesos quantizados utilizados na forward prop
        grad_input = tf.matmul(grad_output, self.qw, transpose_b=True) * (qws * qxs * qgos  / qos)             
        
        # scale de grad_output or grad_of_input
        self.grad_output_scale =  0.9 * self.grad_output_scale + 0.1 * tf.reduce_max(tf.abs(grad_input))
        qgis = quantize_po2(self.grad_output_scale)
        grad_input = grad_input / qgis

        # quantiza o gradiente
        grad_input = quantize(grad_input, stochastic_round=True, stochastic_zero=False)
                
        # calcula o gradiente dos pesos. self.inputs é a entrada dessa camada na etapa de forward prop. Ela é quantizada.         
        grad_weights = tf.matmul(self.inputs, grad_output, transpose_a=True) * (qxs * qgos / qos) 
        grad_biases = tf.reduce_sum(grad_output, axis=0, keepdims=True) * (qgos / qos) 

        # get the grad w scale
        self.grad_weights_scale = 0.9 * self.grad_weights_scale + 0.1 * tf.reduce_max(tf.abs(grad_weights))       
        qgws = quantize_po2(self.grad_weights_scale)
        self.gws_hist.append(qgws)

        # get the grad b scale
        self.grad_bias_scale = 0.9 * self.grad_bias_scale + 0.1 * tf.reduce_max(tf.abs(grad_biases))        
        qgbs = quantize_po2(self.grad_bias_scale)
        self.gbs_hist.append(qgbs)

        # scale the grad
        grad_weights /= qgws
        grad_biases /= qgbs
        
        # quantize the grad
        qgw = quantize(grad_weights, True, stochastic_zero=False)
        qgb = quantize(grad_biases, True, stochastic_zero=False)        


        #################### ETAPA DE ATUALIZAÇÃO DOS PESOS #######################                        

        # weight scaling
        self.qw = self.qw * qws
        # gradient scaling
        qgw = qgw * qgws        
        # weight updating
        self.qw = self.qw - learning_rate * qgw
        
        # bias scaling
        self.qb = self.qb * (qws * qxs) 
        # bias gradient scaling
        qgb = qgb * qgbs
        # bias updating
        self.qb = self.qb - learning_rate * qgb
        
        ############################################################################
        # ############ ETAPA DE CLIP, ESCALA E QUANTIZAÇÃO ################

        # atribui a weights. Weights será escalado e quantizado durante inferência
        w = self.qw 
        w = tf.clip_by_value(w, -7, 7)         
        b = self.qb
        b = tf.clip_by_value(b, -127., 127.) 
            
        # colocar quantização aqui e remover do forward
        # descobre a escala dos pesos com base no valor máximo
        self.weights_scale = 0.9*self.weights_scale + 0.1*tf.reduce_max(tf.abs(w))
        qws = quantize_po2(self.weights_scale)
        self.ws_hist.append(qws)
                
        # escala os pesos 
        w = w / qws
        
        # escala os biases 
        self.bs_hist.append(qws * qxs)
        b = b / (qws * qxs)
                
        # quantiza peesos e bias
        self.qw = quantize(w, True, True)
        self.qb = quantize(b, True, False)
               
        
        return grad_input
    
    
    def reset_weights(self):
        w = tf.constant(np.random.randn(self.input_size, self.output_size) * np.sqrt(2/self.input_size), tf.float32)
        self.weights_scale = tf.reduce_max(tf.abs(w))
        self.qw = quantize(w/self.weights_scale, True, True)
        
        b = tf.zeros((1, self.output_size))        
        self.qb = quantize(b, True, False) # quantized bias