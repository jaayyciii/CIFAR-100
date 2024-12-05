
# CIFAR100 dataset
#   100 Fine Classes, Grouped in 20 Coarse Class [Label 0-19]
#   Each Fine Class: 500 TRAIN Images and 100 TEST Images

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

import numpy as np
# import sys
import matplotlib.pyplot as plt
import pandas as pd
import math

# Extracting CIFAR100 Images
(xF_train, yF_train), (xF_test, yF_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
(xC_train, yC_train), (xC_test, yC_test) = tf.keras.datasets.cifar100.load_data(label_mode='coarse')

print("Hello")
print(xC_train)
