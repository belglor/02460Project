import numpy as np
import tensorflow as tf

# Import MINST data
import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

