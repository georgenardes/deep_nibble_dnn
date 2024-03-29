import numpy as np
import tensorflow as tf

def stochastic_rounding(x):
    s = tf.sign(x)
    abs_x = tf.abs(x)

    # takes the integer part of the number
    x_int = tf.floor(abs_x)
    
    # takes the fractional part
    x_frac = tf.abs(abs_x - x_int)

    # generate a random number    
    rng = tf.random.uniform(x_int.shape, dtype=tf.float32)

    # if the frac is grater... for positive cases
    rounded_x = tf.where(x_frac > rng, x_int + 1, x_int)
    
    # give back the signal
    rounded_x = rounded_x * s

    return rounded_x


def quantize(x, stochastic_round = True, stochastic_zero = True):
    """ exponentiation and quantization function """

    # just to avoid numerical problems
    eps = tf.constant(1e-38, dtype=tf.float32)
    log2 = tf.cast(tf.math.log(tf.constant(2, tf.float32)), tf.float32)

    # extract the signal
    s = tf.sign(x)
    
    # takes the abs
    abs_x = tf.abs(x)

    cliped_abs_x = tf.where(abs_x < eps, eps, abs_x) # clip the min value of abs. (this is just for avoid numercal problems)
    cliped_abs_x = tf.where(cliped_abs_x > 1, 1, cliped_abs_x) # clip the max value of DN 

    # gets the exponent with base 2
    exp = tf.math.log(cliped_abs_x)/log2

    

    # round to nearest and cast to int (use stochastic rounding)
    if stochastic_round:
        round_exp = stochastic_rounding(exp)
    else:
        round_exp = tf.round(exp)

    
    if stochastic_zero:
        ###################
        # stochastic zero
        # detect underflow
        underflow = tf.where(round_exp < -7., True, False)
        # clip expoent in -7
        clip_exp = tf.where(underflow, -7., round_exp)    
        # randomize the signal
        s = tf.where(tf.logical_and(tf.random.uniform(round_exp.shape, 0, 1) < 0.5, underflow), -s, s) 
        # convert to float32 again
        qx = s * tf.pow(tf.constant(2, tf.float32), clip_exp)
        ###################
    else:
        ###################
        # fixed zero    
        # detect underflow
        underflow = tf.where(round_exp < -7, True, False)
        
        # convert to float32 again
        qx = s * tf.pow(2., round_exp)
        
        # fixed zero
        qx = tf.where(underflow, 0, qx)

    return qx




def quantize_po2(x):
    """ x must be positive and greater than 1e-38"""

    eps = tf.constant(1e-38, dtype=tf.float32)
    log2 = tf.cast(tf.math.log(2.), tf.float32)

    # clip the value to be quantized to avoid numerical under/overflow
    cx = tf.where(x < eps, eps, x)    

    qx = tf.pow(2., tf.math.ceil(tf.math.log(cx)/log2))
    
    return qx # bypass