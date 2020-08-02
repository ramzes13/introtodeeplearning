import os
import tensorflow as tf
import mitdeeplearning as mdl
import numpy as np
import matplotlib.pyplot as plt

# disable tf cpu wawnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant(15)
b = tf.constant(61)
# a, b = 1.5, 2.5
def func(a, b):
    c = a + b
    d = b - 1
    e = c * d
    return e

e_out = func(a,b)
print(e_out)

